#!/usr/bin/env python
"""
Разбор взрыва лосса в ldm_v1: данные, веса и внимание на сохраненных чекпоинтах.

  данные  : NaN, inf и выбросы в латентах датасета
  веса    : нормы q и k каждого внимания, всего блока и выходного слоя
  внимание: максимум логитов и энтропия (в долях от log N) на реальных латентах при нескольких t
  лосс    : eps-MSE на одном и том же батче

Запуск: .venv/bin/python tools/diag_ldm.py runs/ldm_v1 ~/DiffusionLearn_data/pretrain_v1
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import UNet
from models.unet import AttnBlock
from train import VPSDE

run, data = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((run / "config.json").read_text())
dev = "cuda"

# ----------------------------------------------------------------------------- данные
shards = sorted((data / "latents").glob("shard_*.npy"))
mx, bad, n = [], 0, 0
for p in shards:
    z = np.load(p, mmap_mode="r")
    for s in range(0, len(z), 2048):
        b = torch.from_numpy(np.asarray(z[s:s + 2048])).float()
        bad += (~torch.isfinite(b)).flatten(1).any(1).sum().item()
        mx.append(b.flatten(1).abs().max(1).values * cfg["scale"])
        n += len(b)
mx = torch.cat(mx)
q = torch.quantile(mx, torch.tensor([0.5, 0.99, 0.999, 1.0])).tolist()
print(f"латенты: {n} кадров, с NaN или inf {bad}, max|z| по кадрам после scale: медиана {q[0]:.2f}, "
      f"99% {q[1]:.2f}, 99.9% {q[2]:.2f}, максимум {q[3]:.2f}")

z_all = torch.from_numpy(np.concatenate([np.load(p, mmap_mode="r")[:40] for p in shards]))
g = torch.Generator().manual_seed(0)
x0 = z_all[torch.randperm(len(z_all), generator=g)[:64], 0].float().to(dev) * cfg["scale"]
eps = torch.randn(x0.shape, generator=g).to(dev)


# ----------------------------------------------------------------------------- чекпоинты
def build():
    return UNet(img_size=64, in_ch=4, base=cfg["base"], mults=[int(v) for v in cfg["mults"].split(",")],
                num_res=cfg["num_res"], attn_res=[int(v) for v in cfg["attn_res"].split(",")],
                dropout=cfg["dropout"], norm_groups=32, head_ch=cfg.get("head_ch") or None, qk_norm=cfg.get("qk_norm", False))


ckpts = []
for p in sorted(run.glob("ema_*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ckpts.append((f"ema {ck['step']}", ck["ema"]))
last = torch.load(run / "ckpt/last.ckpt", map_location="cpu", weights_only=False, mmap=True)
sd = last["state_dict"]
ckpts.append((f"raw {last['global_step']}", {k[6:]: v for k, v in sd.items() if k.startswith("model.")}))
ckpts.append((f"ema {last['global_step']}", {k[4:]: v for k, v in sd.items() if k.startswith("ema.")}))

sde = VPSDE(pred=cfg["pred"])
T = [0.05, 0.3, 0.7]
stats = {}


def hook(name):
    def f(mod, inp, out):
        logits = mod.logits(inp[0])
        p = logits.softmax(-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item() / math.log(logits.shape[-1])
        stats.setdefault(name, []).append((logits.max().item(), ent))
    return f


names = None
for label, state in ckpts:
    net = build()
    net.load_state_dict(state)
    net = net.to(dev).eval()
    attn = [(n_, m) for n_, m in net.named_modules() if isinstance(m, AttnBlock)]
    names = [n_ for n_, _ in attn]
    hs = [m.register_forward_hook(hook(n_)) for n_, m in attn]
    stats.clear()
    losses = []
    with torch.no_grad():
        for t in T:
            tt = torch.full((len(x0),), t, device=dev)
            m_, s_ = sde.m(tt).view(-1, 1, 1, 1), sde.sigma(tt).view(-1, 1, 1, 1)
            xt = m_ * x0 + s_ * eps
            out = net(xt, sde.c(tt))
            losses.append(((sde.to_eps(out, xt, tt) - eps) ** 2).mean().item())
    for h in hs:
        h.remove()
    print(f"\n=== {label}: eps-MSE при t = {T}: " + " ".join(f"{v:.4f}" for v in losses))
    print(f"{'слой':28s} {'|Wq|':>7s} {'|Wk|':>7s} {'|Wv|':>7s} {'|proj|':>7s}  " + "  ".join(f"max/ent t={t}" for t in T))
    for n_, m in attn:
        w = m.qkv.weight.detach().float().flatten(1)
        C = w.shape[0] // 3
        wq, wk, wv = w[:C].norm().item(), w[C:2 * C].norm().item(), w[2 * C:].norm().item()
        row = "  ".join(f"{a:7.1f}/{e:.2f}" for a, e in stats[n_])
        print(f"{n_:28s} {wq:7.2f} {wk:7.2f} {wv:7.2f} {m.proj.weight.norm().item():7.2f}  {row}")
    big = sorted(((p.detach().float().abs().max().item(), n_) for n_, p in net.named_parameters()), reverse=True)[:4]
    print("крупнейшие веса: " + ", ".join(f"{n_} {v:.1f}" for v, n_ in big))
    del net
    torch.cuda.empty_cache()
