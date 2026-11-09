#!/usr/bin/env python
"""
Comparison of samplers and autoguidance on the same starting noises. Row = variant, column = noise.

Autoguidance (Karras et al. 2024): the output of the good model is pushed away from the output of the weak one,
  out = bad + w (good - bad),  w = 1 without guidance. Weak model: an early EMA snapshot of the same run.

Launch: .venv/bin/python scripts/eval/compare_samplers.py runs/ldm_v2 [--n 8] [--seed 0]
Result: <run>/compare/samplers_seed<seed>.png and one image per variant.

Custom rows instead of the standard set, all via SDE with --steps steps:
  --rows "runs/ldm_v3/ckpt/last.ckpt" "runs/ldm_v3/ckpt/last.ckpt|runs/ldm_v2/ckpt/last.ckpt|1.5"
  row = a checkpoint or checkpoint|weak model|w. Result: <run>/compare/<--name>_seed<seed>.jpg
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root: terra_diffusion imports without installing
from terra_diffusion.diffusion import VPSDE, sample
from terra_diffusion.models import UNet


class Guided(nn.Module):
    def __init__(self, good, bad, w):
        super().__init__()
        self.good, self.bad, self.w = good, bad, w

    def forward(self, x, c):
        g, b = self.good(x, c), self.bad(x, c)
        return b + self.w * (g - b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side", type=int, default=256, help="image side on the shared sheet")
    p.add_argument("--rows", nargs="*", default=None, help="custom rows, see the description at the top")
    p.add_argument("--steps", type=int, default=250, help="SDE steps for --rows")
    p.add_argument("--name", default="rows", help="sheet name for --rows")
    a = p.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    dev = "cuda"

    cache = {}

    def load(path):
        path = str(path)
        if path not in cache:
            cache[path] = _load(path)
        return cache[path]

    def _load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sd = ck["ema"] if "ema" in ck else {k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
        m = UNet(img_size=64, in_ch=4, base=cfg["base"], mults=[int(v) for v in cfg["mults"].split(",")],
                 num_res=cfg["num_res"], attn_res=[int(v) for v in cfg["attn_res"].split(",")], dropout=cfg["dropout"],
                 norm_groups=32, head_ch=cfg.get("head_ch") or None, qk_norm=cfg.get("qk_norm", False))
        m.load_state_dict(sd)
        return m.to(dev).eval().requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(cfg["vae"], local_files_only=True, low_cpu_mem_usage=False).to(dev).eval()
    sde = VPSDE(pred=cfg["pred"])
    x0 = torch.randn(a.n, 4, 64, 64, generator=torch.Generator().manual_seed(a.seed)).to(dev)
    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    if a.rows:
        variants, sheet_name = [], f"{a.name}_seed{a.seed}"
        for spec in a.rows:
            parts = spec.split("|")
            label = parts[0].replace("runs/", "").replace("/ckpt/last.ckpt", " final")
            if len(parts) == 1:
                variants.append((f"{label}, sde {a.steps}", load(parts[0]), "sde", a.steps, 1.0))
            else:
                g = parts[1].replace("runs/", "").replace("/ckpt/last.ckpt", " final").replace(".pt", "")
                variants.append((f"{label}, ag {g}, w {parts[2]}", Guided(load(parts[0]), load(parts[1]), float(parts[2])),
                                 "sde", a.steps, 1.0))
    else:
        sheet_name = f"samplers_seed{a.seed}"
        good = load(run / "ckpt/last.ckpt")
        weak = {s: load(run / f"ema_{s:07d}.pt") for s in (5000, 10000)}
        variants = [
        ("heun 50, rho 2 (as in training)", good, "heun", 50, 2.0),
        ("heun 50, rho 1", good, "heun", 50, 1.0),
        ("heun 50, rho 3", good, "heun", 50, 3.0),
        ("heun 200, rho 2", good, "heun", 200, 2.0),
        ("sde 100", good, "sde", 100, 1.0),
        ("sde 250", good, "sde", 250, 1.0),
        ("sde 500", good, "sde", 500, 1.0),
        ("sde 1000", good, "sde", 1000, 1.0),
        ("sde 250, autoguidance 5k, w 1.5", Guided(good, weak[5000], 1.5), "sde", 250, 1.0),
        ("sde 250, autoguidance 5k, w 2.5", Guided(good, weak[5000], 2.5), "sde", 250, 1.0),
        ("sde 250, autoguidance 10k, w 2", Guided(good, weak[10000], 2.0), "sde", 250, 1.0),
        ("heun 50, autoguidance 10k, w 2", Guided(good, weak[10000], 2.0), "heun", 50, 2.0),
    ]
    out = run / "compare"
    out.mkdir(exist_ok=True)
    S, L = a.side, 300
    sheet = Image.new("RGB", (L + a.n * S, len(variants) * S), (255, 255, 255))
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)   # the default PIL font has no Cyrillic
    for r, (name, net, method, steps, rho) in enumerate(variants):
        t0 = time.time()
        torch.manual_seed(a.seed)                                  # the SDE noise is the same for all variants
        with torch.no_grad():
            z = sample(net, sde, x0, steps=steps, rho=rho, method=method, amp_ctx=bf16, clip=False)
            x = torch.cat([vae.decode(z[i:i + 4].float() / cfg["scale"]).sample for i in range(0, len(z), 4)])
        x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        row = Image.new("RGB", (a.n * 512, 512))
        for c, im in enumerate(x):
            row.paste(Image.fromarray(im), (c * 512, 0))
        tag = name.split(" (")[0].replace(", ", "_").replace(" ", "").replace("/", "-")
        row.save(out / f"{tag}_seed{a.seed}.jpg", quality=92)
        sheet.paste(row.resize((a.n * S, S), Image.LANCZOS), (L, r * S))
        ImageDraw.Draw(sheet).multiline_text((10, r * S + S // 2 - 30), name.replace(", ", ",\n"), fill=(0, 0, 0), font=font)
        print(f"{name}: {time.time() - t0:.0f} s, latent std {z.std().item():.2f}", flush=True)
    sheet.save(out / f"{sheet_name}.jpg", quality=90)
    print(f"done: {out / f'{sheet_name}.jpg'}")


if __name__ == "__main__":
    main()
