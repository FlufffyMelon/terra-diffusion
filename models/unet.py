"""
U-Net для диффузионной модели в стиле DDPM.

Вход:  шумная картинка x_t (B, 3, S, S) и уровень шума c (B,), по умолчанию c = log(sigma/m)
Выход: предсказанный шум того же размера (B, 3, S, S)

Энкодер уменьшает разрешение в 2 раза на каждом уровне, декодер увеличивает обратно,
skip-соединения склеивают выходы энкодера со входами декодера по каналам.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(ch, groups=32):
    """GroupNorm. Внимание: при groups = ch это нормировка каждого канала отдельно, и сеть
    теряет средний уровень каждого канала, а с ним и средний цвет картинки."""
    return nn.GroupNorm(math.gcd(groups, ch), ch)


class NoiseEmbedding(nn.Module):
    """Синусы и косинусы от уровня шума c с гладкими частотами, затем MLP."""

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
    """GN -> SiLU -> Conv3x3 -> + сдвиг от шума -> GN -> SiLU -> Dropout -> Conv3x3, плюс обход."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout, groups=32):
        super().__init__()
        self.norm1 = gn(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = gn(out_ch, groups)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)  # блок стартует как тождественный
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """
    Self-attention по всем точкам карты признаков.
      head_ch : каналов на голову, None = одна голова на все каналы, как было раньше
      qk_norm : RMSNorm на q и k каждой головы. Без нее логиты q.k растут вместе с весами,
                и softmax схлопывается в одну точку (так сломался ldm_v1).
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
        """(B, C, H, W) -> q, k, v, каждый (B, heads, N, C / heads). Канал c головы h это h * d + c."""
        B, C, H, W = x.shape
        # contiguous: flash attention берет только тензоры с непрерывной последней осью,
        # иначе SDPA строит полную матрицу внимания N x N на каждую голову
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).contiguous().unbind(0)
        if self.q_norm is not None:
            # нормировка в float32, обратно в тип v, потому что SDPA требует один тип у q, k, v
            q, k = self.q_norm(q.float()).to(v.dtype), self.k_norm(k.float()).to(v.dtype)
        return q, k, v

    def logits(self, x):
        """Логиты внимания (B, heads, N, N) в float32, для диагностики."""
        q, k, _ = self.qkv_heads(x)
        return q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_heads(x)
        h = F.scaled_dot_product_attention(q, k, v)                          # (B, heads, N, d)
        h = h.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(h)


class Stage(nn.Module):
    """Residual-блок и, если нужно, внимание после него."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout, attn, groups=32, head_ch=None, qk_norm=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim, dropout, groups)
        self.attn = AttnBlock(out_ch, groups, head_ch, qk_norm) if attn else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.res(x, emb))


class Downsample(nn.Module):
    """Свертка 3x3 с шагом 2: разрешение / 2, каналы те же."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Повтор каждой точки в квадрат 2x2, затем свертка 3x3."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """
    img_size : размер входной картинки (нужен, чтобы знать, где ставить внимание)
    base     : число каналов на верхнем уровне
    mults    : множители каналов по уровням; len(mults) = число уровней,
               при img_size = 64 и 4 уровнях разрешения 64, 32, 16, 8
    num_res  : residual-блоков на уровень в энкодере (в декодере на один больше)
    attn_res : разрешения, на которых после каждого res-блока стоит внимание
    dropout  : доля зануляемых признаков внутри res-блоков
    norm_groups : число групп в GroupNorm (не больше, чем каналов). В DDPM на CIFAR 32 группы
               на 128 каналов, по 4 канала в группе. 32 группы на 32 канала ломают средний цвет.
    head_ch  : каналов на голову внимания, None = одна голова (старые модели)
    qk_norm  : RMSNorm на q и k во всех блоках внимания
    """

    def __init__(self, img_size=64, in_ch=3, base=64, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.1,
                 norm_groups=32, head_ch=None, qk_norm=False):
        super().__init__()
        self.config = dict(img_size=img_size, in_ch=in_ch, base=base, mults=tuple(mults),
                           num_res=num_res, attn_res=tuple(attn_res), dropout=dropout, norm_groups=norm_groups,
                           head_ch=head_ch, qk_norm=qk_norm)
        g = norm_groups
        att = dict(head_ch=head_ch, qk_norm=qk_norm)
        emb_dim = 4 * base
        self.embed = NoiseEmbedding(base, emb_dim)
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        # энкодер: запоминаем число каналов каждого skip-соединения
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

        # декодер: на каждом блоке приклеиваем skip с того же разрешения
        self.up = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(mults))):
            out = base * mult
            for _ in range(num_res + 1):
                self.up.append(Stage(ch + skip_ch.pop(), out, emb_dim, dropout, attn=res in attn_res, groups=g, **att))
                ch = out
            if lvl != 0:
                self.up.append(Upsample(ch))
                res *= 2

        self.norm_out = gn(ch, g)  # последняя нормировка перед выходом: через нее идет весь ответ сети
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)  # в начале сеть выдает 0, лосс = 1
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, c):
        emb = self.embed(c)
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
