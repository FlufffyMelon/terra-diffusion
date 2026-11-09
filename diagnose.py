#!/usr/bin/env python
"""
Diagnostics of a trained model without interfering with training.

1) denoise.png  : real photos -> noise up to time t -> one network prediction -> denoiser
                  D = (x_t - sigma * eps_theta) / m.  Rows: clean photos, then for each t a pair of rows
                  (noisy image, reconstructed one).
2) samplers.png : the same starting noise, different ways of solving the reverse process.
                  Each row = one sampler (the order is printed to the console).

Example:
  srun --jobid=139160 --overlap -n 1 -c 2 .venv/bin/python diagnose.py \
      --run runs/cat64_fast --data-dir Terra --exclude-file exclude.txt
"""
import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from models import model_from_checkpoint
from train import sde_from_cfg, center_crops, load_images, sample, save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None, help="default run/last.pt")
    p.add_argument("--data-dir", type=str, default="Terra")
    p.add_argument("--exclude-file", type=str, default=None)
    p.add_argument("--n", type=int, default=8, help="images per row")
    a = p.parse_args()

    run = Path(a.run)
    ck = torch.load(a.ckpt or run / "last.pt", map_location="cpu", weights_only=False)
    cfg = ck["args"]
    device = "cuda"

    model = model_from_checkpoint(ck, weights="ema")
    model = model.to(device).eval()
    sde = sde_from_cfg(cfg)
    S = cfg["img_size"]
    step = ck["step"]
    print(f"checkpoint: step {step}")

    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    fp32 = nullcontext

    # ------------------------------------------------------------------ 1. one-step denoising
    exclude = set()
    if a.exclude_file:
        for line in Path(a.exclude_file).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                exclude.add(line)
    _, imgs = load_images(a.data_dir, cfg["pre_size"], exclude)
    idx = torch.linspace(0, len(imgs) - 1, a.n).long().tolist()   # evenly across the dataset
    x0 = center_crops([imgs[i] for i in idx], S).to(device)

    g = torch.Generator(device=device).manual_seed(0)
    eps = torch.randn(x0.shape, generator=g, device=device)
    rows = [x0]
    print(f"\none-step denoising (fp32, network input: {sde.cond}):")
    print(f"  x0-MSE = (sigma/m)^2 * eps-MSE, error of the reconstructed image.")
    print(f"  For comparison: if one always answers with the mean photo, x0-MSE = {x0.var(0).mean().item():.3f}")
    with torch.no_grad():
        for t_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            t = torch.full((a.n,), t_val, device=device)
            m = sde.m(t).view(-1, 1, 1, 1)
            s = sde.sigma(t).view(-1, 1, 1, 1)
            xt = m * x0 + s * eps
            e = sde.to_eps(model(xt, sde.c(t)).float(), xt, t)
            D_raw = (xt - s * e) / m
            D = D_raw.clamp(-1, 1)
            mse = ((e - eps) ** 2).mean().item()
            x0_mse = ((D_raw - x0) ** 2).mean().item()
            dc = ((D_raw.mean((2, 3)) - x0.mean((2, 3))) ** 2).mean().sqrt().item()  # mean color error
            print(f"  t={t_val:<4}  sigma={s[0].item():.3f}  m={m[0].item():.3f}  c={sde.c(t)[0].item():6.3f}  "
                  f"eps-MSE={mse:.4f}  x0-MSE={x0_mse:.3f}  mean color error={dc:.3f}")
            rows += [xt.clamp(-1, 1), D]
    save_grid(torch.cat(rows), run / f"diag_denoise_step{step}.png", nrow=a.n)

    # ------------------------------------------------------------------ 2. sampler comparison
    noise = torch.randn(a.n, 3, S, S, generator=torch.Generator().manual_seed(1234)).to(device)
    configs = [
        ("heun 50 steps, bf16 (as in training)", "heun", 50, 2.0, bf16),
        ("heun 50 steps, fp32", "heun", 50, 2.0, fp32),
        ("heun 200 steps, fp32", "heun", 200, 2.0, fp32),
        ("euler 500 steps, fp32", "euler", 500, 1.0, fp32),
        ("SDE 1000 steps, fp32", "sde", 1000, 1.0, fp32),
    ]
    rows = []
    print("\nsamplers (rows of samplers.png from top to bottom):")
    for name, method, steps, rho, ctx in configs:
        t0 = time.time()
        torch.manual_seed(0)  # the same noise inside the SDE between runs
        x = sample(model, sde, noise, steps=steps, rho=rho, method=method, amp_ctx=ctx)
        rows.append(x)
        print(f"  {name:40s} {time.time() - t0:5.1f} s")
    save_grid(torch.cat(rows), run / f"diag_samplers_step{step}.png", nrow=a.n)
    print(f"\nimages saved to {run}")


if __name__ == "__main__":
    main()
