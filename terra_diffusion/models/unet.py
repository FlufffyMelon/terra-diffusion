"""
U-Net for a DDPM-style diffusion model.

Input:  noisy image x_t (B, 3, S, S) and noise level c (B,), by default c = log(sigma/m)
Output: predicted noise of the same size (B, 3, S, S)

The encoder halves the resolution at each level, the decoder increases it back,
skip connections concatenate the encoder outputs with the decoder inputs along channels.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(ch, groups=32):
    """GroupNorm. Note: with groups = ch this normalizes each channel separately, and the network
    loses the mean level of each channel, and with it the mean color of the image."""
    return nn.GroupNorm(math.gcd(groups, ch), ch)


class NoiseEmbedding(nn.Module):
    """Sines and cosines of the noise level c with smooth frequencies, then an MLP."""

    def __init__(self, dim, out_dim, f_min=1.0, f_max=8.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(f_min), math.log(f_max), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * half, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, c):
        a = c.float()[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=1))


class ResBlock(nn.Module):
    """GN -> SiLU -> Conv3x3 -> + noise shift -> GN -> SiLU -> Dropout -> Conv3x3, plus a skip path."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout, groups=32):
        super().__init__()
        self.norm1 = gn(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = gn(out_ch, groups)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)  # the block starts as identity
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """
    Self-attention over all points of the feature map.
      head_ch : channels per head, None = one head over all channels, as before
      qk_norm : RMSNorm on q and k of each head. Without it the q.k logits grow with the weights,
                and softmax collapses to a single point (this is how ldm_v1 broke).
    """

    def __init__(self, ch, groups=32, head_ch=None, qk_norm=False):
        super().__init__()
        self.norm = gn(ch, groups)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.heads = ch // head_ch if head_ch else 1
        d = ch // self.heads
        self.q_norm = nn.RMSNorm(d, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(d, eps=1e-6) if qk_norm else None

    def qkv_heads(self, x):
        """(B, C, H, W) -> q, k, v, each (B, heads, N, C / heads). Channel c of head h is h * d + c."""
        B, C, H, W = x.shape
        # contiguous: flash attention only takes tensors with a contiguous last axis,
        # otherwise SDPA builds the full N x N attention matrix for each head
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).contiguous().unbind(0)
        if self.q_norm is not None:
            # normalization in float32, cast back to the dtype of v, because SDPA requires the same dtype for q, k, v
            q, k = self.q_norm(q.float()).to(v.dtype), self.k_norm(k.float()).to(v.dtype)
        return q, k, v

    def logits(self, x):
        """Attention logits (B, heads, N, N) in float32, for diagnostics."""
        q, k, _ = self.qkv_heads(x)
        return q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_heads(x)
        h = F.scaled_dot_product_attention(q, k, v)                          # (B, heads, N, d)
        h = h.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(h)


class Stage(nn.Module):
    """Residual block and, if needed, attention after it."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout, attn, groups=32, head_ch=None, qk_norm=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, dropout, groups)
        self.attn = AttnBlock(out_ch, groups, head_ch, qk_norm) if attn else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.res(x, emb))


class Downsample(nn.Module):
    """3x3 convolution with stride 2: resolution / 2, same channels."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Repeat each point into a 2x2 square, then a 3x3 convolution."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """
    img_size : size of the input image (needed to know where to put attention)
    base     : number of channels at the top level
    mults    : channel multipliers per level; len(mults) = number of levels,
               for img_size = 64 and 4 levels the resolutions are 64, 32, 16, 8
    num_res  : residual blocks per level in the encoder (the decoder has one more)
    attn_res : resolutions at which attention follows each res block
    dropout  : fraction of features zeroed inside res blocks
    norm_groups : number of GroupNorm groups (at most the number of channels). In DDPM on CIFAR 32 groups
               for 128 channels, 4 channels per group. 32 groups on 32 channels break the mean color.
    head_ch  : channels per attention head, None = one head (old models)
    qk_norm  : RMSNorm on q and k in all attention blocks
    cond_sizes : number of values of each condition, e.g. (2, 100): who (another cat, Terra) and type (DINOv2 cluster).
               Each condition has its own embedding table with an extra "empty" value (index size) for CFG.
               The embeddings are added to the noise embedding and start from zero: fine-tuning from the weights of a model
               without conditions starts exactly from its behavior.
    """

    def __init__(self, img_size=64, in_ch=3, base=64, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.1,
                 norm_groups=32, head_ch=None, qk_norm=False, cond_sizes=()):
        super().__init__()
        self.config = dict(img_size=img_size, in_ch=in_ch, base=base, mults=tuple(mults),
                           num_res=num_res, attn_res=tuple(attn_res), dropout=dropout, norm_groups=norm_groups,
                           head_ch=head_ch, qk_norm=qk_norm, cond_sizes=tuple(cond_sizes))
        g = norm_groups
        att = dict(head_ch=head_ch, qk_norm=qk_norm)
        emb_dim = 4 * base
        self.embed = NoiseEmbedding(base, emb_dim)
        self.cond = nn.ModuleList(nn.Embedding(n + 1, emb_dim) for n in cond_sizes)
        for e in self.cond:
            nn.init.zeros_(e.weight)
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        # encoder: remember the number of channels of each skip connection
        self.down = nn.ModuleList()
        skip_ch = [base]
        ch, res = base, img_size
        for lvl, mult in enumerate(mults):
            out = base * mult
            for _ in range(num_res):
                self.down.append(Stage(ch, out, emb_dim, dropout, attn=res in attn_res, groups=g, **att))
                ch = out
                skip_ch.append(ch)
            if lvl != len(mults) - 1:
                self.down.append(Downsample(ch))
                skip_ch.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout, g)
        self.mid_attn = AttnBlock(ch, g, **att)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout, g)

        # decoder: at each block concatenate the skip from the same resolution
        self.up = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(mults))):
            out = base * mult
            for _ in range(num_res + 1):
                self.up.append(Stage(ch + skip_ch.pop(), out, emb_dim, dropout, attn=res in attn_res, groups=g, **att))
                ch = out
            if lvl != 0:
                self.up.append(Upsample(ch))
                res *= 2

        self.norm_out = gn(ch, g)  # the last normalization before the output: the whole network response passes through it
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # at the start the network outputs 0, loss = 1
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, c, y=None):
        """y : (B, number of conditions) long, value size = "empty". None = all conditions empty."""
        emb = self.embed(c)
        if len(self.cond):
            if y is None:
                y = torch.tensor([e.num_embeddings - 1 for e in self.cond], device=x.device).expand(len(x), -1)
            for k, e in enumerate(self.cond):
                emb = emb + e(y[:, k])
        h = self.conv_in(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, Stage) else layer(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, Stage):
                h = layer(torch.cat([h, hs.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.conv_out(F.silu(self.norm_out(h)))
