#!/usr/bin/env python
"""
Latent diffusion quality: FD-DINOv2 and KID between samples and real frames.

  FD-DINOv2 : Frechet distance between Gaussians of DINOv2 ViT-L/14 features (CLS after LayerNorm, input 224),
              as in Stein et al. 2023. Sensitive to overall shift and spread, but biased at small N:
              compare only at the same number of samples.
  KID       : unbiased estimate of MMD^2 with a polynomial kernel on Inception-v3 features (as in experiments/pixel64/memorization.py),
              works with a thousand samples too, gives the spread over subsets.
  For free  : FID on Inception and KID on DINOv2, the features are computed anyway.

Reference: --n-real fit_1x1 frames from the manifest, one crop each with the same rule as in training (box inside).
Two anchor rows to understand the scale:
  real            : other --n-gen real frames. The lower bound of the metrics at this N.
  reconstruction  : the same frames through the VAE. The best one can get with this VAE.

Rows: a checkpoint or checkpoint|weak model|w (autoguidance), all via SDE with --steps steps.
  .venv/bin/python scripts/eval/eval_metrics.py --dataset ~/DiffusionLearn_data/pretrain_v1 --out runs/eval/v2_v3 \\
      --rows runs/ldm_v3/ckpt/last.ckpt "runs/ldm_v3/ckpt/last.ckpt|runs/ldm_v2/ckpt/last.ckpt|2"
In --out: metrics.csv (one row per variant, appended as they finish), feats/*.npz, grids/*.jpg.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root: terra_diffusion imports without installing
from terra_diffusion.checkpoints import load_unet
from terra_diffusion.data import FrameCrops, read_manifest
from terra_diffusion.diffusion import VPSDE, sample
from terra_diffusion.guidance import Guided
from terra_diffusion.metrics import fd, inception_feats, kid, load_inception

DINO = "facebook/dinov2-large"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--n-real", type=int, default=10000)
    p.add_argument("--n-gen", type=int, default=5000)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-baselines", action="store_true", help='without the "real" and "VAE reconstruction" rows')
    return p.parse_args()


class Features:
    """DINOv2 and Inception for images (B, 3, S, S) in [-1, 1] on the GPU."""

    def __init__(self, dev):
        self.dino = AutoModel.from_pretrained(DINO, local_files_only=True).to(dev).eval()
        self.inc = load_inception(dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        b = (x.float().clamp(-1, 1) + 1) / 2
        b = F.interpolate(b, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        d = self.dino(pixel_values=(b - self.mean) / self.std).pooler_output.double().cpu()
        return d, inception_feats(self.inc, x)


def real_loader(rows, root, idx, a):
    ds = Subset(FrameCrops(rows, root, size=512, crops=1, seed=a.seed), idx)
    return DataLoader(ds, batch_size=a.batch, num_workers=a.workers, prefetch_factor=4)


def main():
    a = parse_args()
    dev = "cuda"
    out = Path(a.out)
    (out / "feats").mkdir(parents=True, exist_ok=True)
    (out / "grids").mkdir(exist_ok=True)
    root = Path(a.dataset)
    rows = read_manifest(root / "manifest.jsonl")
    perm = np.random.default_rng(a.seed).permutation(len(rows))
    real_idx, base_idx = perm[:a.n_real].tolist(), perm[a.n_real:a.n_real + a.n_gen].tolist()
    feat = Features(dev)
    t_start = time.time()

    # ------------------------------------------------------------- reference
    ref_path = out / "feats" / "real.npz"
    if ref_path.exists():
        z = np.load(ref_path)
        ref_d, ref_i = torch.from_numpy(z["dino"]), torch.from_numpy(z["inc"])
    else:
        fd_, fi_ = [], []
        for x, _, ok in real_loader(rows, root, real_idx, a):
            d, i = feat(x[ok, 0].to(dev))
            fd_.append(d); fi_.append(i)
        ref_d, ref_i = torch.cat(fd_), torch.cat(fi_)
        np.savez(ref_path, dino=ref_d.numpy(), inc=ref_i.numpy())
    print(f"reference: {len(ref_d)} frames, {time.time() - t_start:.0f} s", flush=True)

    csv_path = out / "metrics.csv"
    new = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["variant", "n", "fd_dinov2", "kid_inception_x1000", "kid_inception_std_x1000", "fid_inception",
                    "kid_dinov2_x1000", "seconds", "images_per_second"])

    def report(name, d, i, sec):
        k, ks = kid(ref_i, i)
        kd, _ = kid(ref_d, d)
        r = [name, len(d), f"{fd(ref_d, d):.2f}", f"{k * 1e3:.3f}", f"{ks * 1e3:.3f}", f"{fd(ref_i, i):.2f}", f"{kd * 1e3:.3f}",
             f"{sec:.0f}", f"{len(d) / max(sec, 1e-9):.2f}"]
        w.writerow(r)
        f.flush()
        print(f"{name}: FD-DINOv2 {r[2]} | KID x1000 {r[3]} ± {r[4]} | FID {r[5]} | KID-DINOv2 x1000 {r[6]} | {r[7]} s", flush=True)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True, low_cpu_mem_usage=False).to(dev).eval()

    # ------------------------------------------------------------- anchor rows
    if not a.no_baselines:
        t0 = time.time()
        acc = {"real": ([], []), "VAE reconstruction": ([], [])}
        for x, _, ok in real_loader(rows, root, base_idx, a):
            x = x[ok, 0].to(dev)
            with torch.no_grad():
                # in chunks of 16: the VAE encoder does not fit 64 images of 512x512 into 24 GB
                rec = torch.cat([vae.decode(vae.encode(x[j:j + 16]).latent_dist.mean).sample for j in range(0, len(x), 16)])
            for key, img in (("real", x), ("VAE reconstruction", rec)):
                d, i = feat(img)
                acc[key][0].append(d); acc[key][1].append(i)
        sec = time.time() - t0
        for key, (d, i) in acc.items():
            report(key, torch.cat(d), torch.cat(i), sec)

    # ------------------------------------------------------------- variants
    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    for spec in a.rows:
        parts = spec.split("|")
        good, cfg, gname = load_unet(parts[0], dev)
        net, name = good, f"{gname}, sde {a.steps}"
        if len(parts) == 3:
            bad, _, bname = load_unet(parts[1], dev)
            net, name = Guided(good, bad, float(parts[2])), f"{gname}, ag {bname}, w {parts[2]}, sde {a.steps}"
        sde = VPSDE(pred=cfg["pred"])
        t0 = time.time()
        fd_, fi_ = [], []
        for b in range(0, a.n_gen, a.batch):
            n = min(a.batch, a.n_gen - b)
            s = a.seed * 1_000_003 + b                                  # the same noises for all variants
            x0 = torch.randn(n, 4, 64, 64, generator=torch.Generator().manual_seed(s)).to(dev)
            torch.manual_seed(s)
            with torch.no_grad():
                zz = sample(net, sde, x0, steps=a.steps, rho=1.0, method="sde", amp_ctx=bf16, clip=False)
                img = torch.cat([vae.decode(zz[j:j + 16].float() / cfg["scale"]).sample for j in range(0, n, 16)])
            if b == 0:
                save_image((img[:32].clamp(-1, 1) + 1) / 2, out / "grids" / f"{name.replace(', ', '_').replace(' ', '')}.jpg", nrow=8)
            d, i = feat(img)
            fd_.append(d); fi_.append(i)
            if b // a.batch % 10 == 0:
                done = b + n
                print(f"  {name}: {done} of {a.n_gen}, {done / (time.time() - t0):.2f} images/s", flush=True)
        d, i = torch.cat(fd_), torch.cat(fi_)
        np.savez(out / "feats" / f"{name.replace(', ', '_').replace(' ', '')}.npz", dino=d.numpy(), inc=i.numpy())
        report(name, d, i, time.time() - t0)
        del net, good
        torch.cuda.empty_cache()
    print(f"done in {(time.time() - t_start) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
