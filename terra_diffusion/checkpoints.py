"""
Loading trained latent diffusion models from a run folder: EMA weights from ckpt/last.ckpt or ema_*.pt,
the architecture from the run's config.json.
"""
import json
from pathlib import Path

import torch

from .models import UNet


def load_unet(path, dev):
    path = Path(path)
    run = path.parent.parent if path.parent.name == "ckpt" else path.parent
    cfg = json.loads((run / "config.json").read_text())
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["ema"] if "ema" in ck else {k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    m = UNet(img_size=64, in_ch=4, base=cfg["base"], mults=[int(v) for v in cfg["mults"].split(",")], num_res=cfg["num_res"],
             attn_res=[int(v) for v in cfg["attn_res"].split(",")], dropout=cfg["dropout"], norm_groups=32,
             head_ch=cfg.get("head_ch") or None, qk_norm=cfg.get("qk_norm", False), cond_sizes=cfg.get("cond_sizes", []))
    m.load_state_dict(sd)
    return m.to(dev).eval().requires_grad_(False), cfg, f"{run.name}-{ck.get('step', ck.get('global_step'))}"
