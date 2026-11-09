#!/usr/bin/env python
"""
Computes latents for the frames in manifest.jsonl: --crops random crops per frame, the cat box is always inside.

<dataset>/latents/shard_XX.npy holds float16 (frames, crops, 4, 64, 64), shard_XX.json holds the paths of those frames.
--source and --out-dir compute a separate set, e.g. Terra with 64 crops in latents_terra/.
No scale factor is applied: raw latents, CatLatents computes it.
One job per GPU, the work is split into shards by --shard and --nshards. A finished shard is skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root: terra_diffusion imports without installing
from terra_diffusion.data import FrameCrops, read_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--crops", type=int, default=4)
    p.add_argument("--frames", type=int, default=6, help="frames per batch, frames * crops images in total")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--flag", default="fit_1x1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--source", default="", help="only frames from this source, e.g. terra")
    p.add_argument("--out-dir", default="latents", help="folder inside the dataset")
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(a.dataset)
    out = root / a.out_dir
    out.mkdir(exist_ok=True)
    path = out / f"shard_{a.shard:02d}.npy"
    if path.exists():
        print(f"{path} already exists", flush=True)
        return
    rows = [r for r in read_manifest(root / "manifest.jsonl", a.flag) if not a.source or r["source"] == a.source][a.shard::a.nshards]
    rows = rows[:a.limit] if a.limit else rows
    ds = FrameCrops(rows, root, crops=a.crops, seed=a.shard)
    loader = DataLoader(ds, batch_size=a.frames, num_workers=a.workers, prefetch_factor=4)
    vae = AutoencoderKL.from_pretrained(a.vae, local_files_only=True, low_cpu_mem_usage=False).cuda().eval().requires_grad_(False)
    res, names, bad = [], [], 0
    t0 = time.time()
    with torch.inference_mode():
        for b, (x, idx, ok) in enumerate(loader):
            x = x[ok]
            bad += int((~ok).sum())
            if not len(x):
                continue
            n, k = x.shape[:2]
            x = x.flatten(0, 1).cuda(non_blocking=True)
            # in chunks of 16: the VAE encoder does not fit 64 images of 512x512 into 24 GB
            z = torch.cat([vae.encode(x[j:j + 16]).latent_dist.mean for j in range(0, len(x), 16)])
            res.append(z.view(n, k, *z.shape[1:]).half().cpu().numpy())
            names += [rows[i]["path"] for i in idx[ok].tolist()]
            if b % 200 == 0:
                print(f"shard {a.shard}: {len(names)} of {len(rows)}, {len(names) * k / (time.time() - t0):.1f} images/s", flush=True)
    tmp = out / f"shard_{a.shard:02d}.tmp.npy"
    np.save(tmp, np.concatenate(res))
    (out / f"shard_{a.shard:02d}.json").write_text(json.dumps(names, ensure_ascii=False))
    tmp.rename(path)
    print(f"shard {a.shard}: done {len(names)} frames, broken {bad}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
