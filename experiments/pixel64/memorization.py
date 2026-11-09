#!/usr/bin/env python
"""
How much the model copies the training photos.

For each checkpoint we generate N images from the same noise and for each one look for the
nearest training photo among all crop variants (scales, shifts, flips), as during training.

  d1    = distance to the nearest photo,
  d2    = distance to the second nearest (different) photo,
  ratio = d1 / d2.

For a copy, d1 is much smaller than the distance to any other photo, and ratio is small.
A blurry "average" image is close to many photos at once and has ratio near 1,
so the test does not mistake it for a copy (unlike d1 alone).

Calibration: the same for the training photos themselves (center crop) against all the other photos.
This is the level given by real but different pictures.

Quality: KID between Inception-v3 features of the samples and of the real photos (lower = better).
Note that a model that simply copies photos gets a very good KID,
so KID is read only together with ratio.

Runs trained with --boxes are checked with the same --boxes: crop variants, calibration and real photos for KID
are taken the same way as during training, i.e. the cat box is always inside the crop.

Example:
  sbatch -p AMD7V12-RTX4500-CCP --gpus=1 -c 8 --mem=48G -t 02:00:00 -o runs/memorization_%j.log \
      --wrap ".venv/bin/python experiments/pixel64/memorization.py --runs runs/cat64_eps runs/cat64_uniq \
              --data-dirs Terra dataset_unique --exclude-files experiments/pixel64/exclude_eps.txt -"
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root: terra_diffusion imports without installing
from terra_diffusion.metrics import build_refs, inception_feats, kid, load_inception, nearest
from terra_diffusion.models import model_from_checkpoint
from train import CatDataset, center_crops, load_boxes, load_images, sample, save_grid, sde_from_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run folders")
    p.add_argument("--data-dirs", nargs="+", required=True, help="one photo folder per run")
    p.add_argument("--exclude-files", nargs="+", default=None,
                   help="one exclusion file per run, '-' if there is nothing to exclude")
    p.add_argument("--n", type=int, default=512, help="how many images to generate per checkpoint")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--method", default="heun", choices=["heun", "euler", "sde"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--scales", default="1.0,0.925,0.85", help="crop scales, as during training")
    p.add_argument("--npos", type=int, default=4, help="shifts along each axis")
    p.add_argument("--boxes", default=None, help="boxes.json: crops as in training with --boxes")
    p.add_argument("--nsides", type=int, default=5, help="crop sides per photo with --boxes")
    p.add_argument("--n-real", type=int, default=4, help="random crops per photo for KID with --boxes")
    p.add_argument("--out", default="runs/memorization")
    p.add_argument("--no-kid", action="store_true", help="do not compute KID (the quality metric)")
    return p.parse_args()


# ----------------------------------------------------------------------------- KID


def read_exclude(path):
    ex = set()
    if path and path != "-":
        for line in Path(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                ex.add(line)
    return ex


def main():
    a = parse_args()
    assert len(a.runs) == len(a.data_dirs), "need one data folder per run"
    ex_files = a.exclude_files or ["-"] * len(a.runs)
    assert len(ex_files) == len(a.runs), "need one exclusion file per run"
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(v) for v in a.scales.split(",")]
    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    inception = None if a.no_kid else load_inception(dev)

    summary = []
    for run, ddir, exf in zip(a.runs, a.data_dirs, ex_files):
        run = Path(run)
        name = run.name
        exclude = read_exclude(exf)

        # ---------- checkpoints: all ema_*.pt and last.pt if its step is not among the snapshots
        cks = [(int(p.stem.split("_")[1]), p) for p in sorted(run.glob("ema_*.pt"))]
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        cfg = last["args"]
        model = model_from_checkpoint(last, weights=None).to(dev).eval()
        if last["step"] not in [s for s, _ in cks]:
            cks.append((last["step"], run / "last.pt"))
        cks.sort()
        del last
        S = cfg["img_size"]
        print(f"\n=== {name}: checkpoints {[s for s, _ in cks]}", flush=True)

        # ---------- training photos and all their crop variants
        t0 = time.time()
        files, imgs = load_images(ddir, cfg["pre_size"], exclude)
        boxes = load_boxes(a.boxes, files, imgs) if a.boxes else None
        refs, owner = build_refs(imgs, S, scales, a.npos, boxes, a.nsides)
        refs = refs.flatten(1).to(dev)
        owner = owner.to(dev)
        n_img = len(imgs)
        print(f"photos {n_img}, crop variants {refs.shape[0]} ({refs.shape[0] / n_img:.0f} per photo), "
              f"{time.time() - t0:.1f} s", flush=True)

        # ---------- calibration on the photos themselves
        real = center_crops(imgs, S)
        if boxes is not None:   # real crops with the same random scheme as in training; the first n_img, one per photo
            torch.manual_seed(2026)
            ds = CatDataset(imgs, S, cfg["min_scale"], repeats=1, boxes=boxes)
            real = torch.stack([ds[k] for _ in range(a.n_real) for k in range(n_img)])
        q = real[:n_img].to(dev).flatten(1)
        b1, b2, _, _ = nearest(q, refs, owner, n_img, exclude_self=torch.arange(n_img, device=dev))
        base_ratio = (b1 / b2).cpu().numpy()
        print(f"real photos: median d1 {b1.median():.1f}, ratio<0.5 {np.mean(base_ratio < 0.5) * 100:.1f}%, "
              f"ratio<0.7 {np.mean(base_ratio < 0.7) * 100:.1f}%", flush=True)
        f_real = None if inception is None else inception_feats(inception, real.to(dev))

        # ---------- generation and neighbor search for each checkpoint
        sde = sde_from_cfg(cfg)
        noise = torch.randn(a.n, 3, S, S, generator=torch.Generator().manual_seed(2026))

        steps, D1, D2, R, NN, KID = [], [], [], [], [], []
        for step, path in cks:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["ema"])
            del ck
            t0 = time.time()
            xs = []
            for i in range(0, a.n, a.batch):
                torch.manual_seed(1000 + i)  # same noise inside the SDE for all checkpoints
                xs.append(sample(model, sde, noise[i:i + a.batch].to(dev), steps=a.steps,
                                 rho=a.rho if a.method != "sde" else 1.0, method=a.method, amp_ctx=bf16))
            x = torch.cat(xs)
            d1, d2, nn_img, best_idx = nearest(x.flatten(1).float(), refs, owner, n_img)
            r = (d1 / d2).cpu().numpy()

            # the 16 most suspicious: the sample and the best crop of the photo side by side
            order = np.argsort(r)[:16]
            pairs = torch.stack([torch.stack([x[k].float(), refs[best_idx[k]].view(3, S, S)]) for k in order]).flatten(0, 1)
            save_grid(pairs, out / f"{name}_step{step:07d}_top16.png", nrow=8)

            steps.append(step); D1.append(d1.cpu().numpy()); D2.append(d2.cpu().numpy())
            R.append(r); NN.append(nn_img.cpu().numpy())
            row = dict(run=name, step=step, median_d1=float(np.median(D1[-1])),
                       ratio_lt_05=float(np.mean(r < 0.5) * 100), ratio_lt_07=float(np.mean(r < 0.7) * 100),
                       unique_nn=int(len(np.unique(NN[-1]))))
            if f_real is not None:
                k_mean, k_std = kid(f_real, inception_feats(inception, x))
                row["kid_x1000"], row["kid_std_x1000"] = k_mean * 1e3, k_std * 1e3
                KID.append(k_mean)
            summary.append(row)
            print(f"step {step:7d} | {time.time() - t0:5.1f} s | median d1 {row['median_d1']:.1f} | "
                  f"ratio<0.5 {row['ratio_lt_05']:.1f}% | ratio<0.7 {row['ratio_lt_07']:.1f}% | "
                  f"distinct nearest photos {row['unique_nn']} of {n_img}"
                  + (f" | KID x1000 {row['kid_x1000']:.2f} ± {row['kid_std_x1000']:.2f}" if f_real is not None else ""),
                  flush=True)

        np.savez(out / f"{name}.npz", steps=np.array(steps), d1=np.array(D1), d2=np.array(D2),
                 ratio=np.array(R), nn_img=np.array(NN), base_d1=b1.cpu().numpy(), base_d2=b2.cpu().numpy(),
                 base_ratio=base_ratio, files=np.array([f.name for f in files]),
                 method=a.method, n_steps=a.steps, kid=np.array(KID))
        del refs, owner, model
        torch.cuda.empty_cache()

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary for k in r)))
        w.writeheader()
        w.writerows(summary)
    print(f"\ndone, results in {out}", flush=True)


if __name__ == "__main__":
    main()
