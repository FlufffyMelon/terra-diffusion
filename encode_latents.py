#!/usr/bin/env python
"""
Считает латенты для кадров из manifest.jsonl: на кадр --crops случайных кропов, бокс кошки всегда внутри.

В <dataset>/latents/shard_XX.npy лежит float16 (кадров, кропов, 4, 64, 64), в shard_XX.json пути этих кадров.
Коэффициент масштаба не применяется: сырые латенты, его считает CatLatents.
Одна задача на один GPU, части делятся по --shard и --nshards. Готовая часть пропускается.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from dataset import FrameCrops, read_manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--crops", type=int, default=4)
    p.add_argument("--frames", type=int, default=6, help="кадров в батче, всего картинок frames * crops")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--flag", default="fit_1x1")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(a.dataset)
    out = root / "latents"
    out.mkdir(exist_ok=True)
    path = out / f"shard_{a.shard:02d}.npy"
    if path.exists():
        print(f"{path} уже есть", flush=True)
        return
    rows = read_manifest(root / "manifest.jsonl", a.flag)[a.shard::a.nshards]
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
            z = vae.encode(x.flatten(0, 1).cuda(non_blocking=True)).latent_dist.mean
            res.append(z.view(n, k, *z.shape[1:]).half().cpu().numpy())
            names += [rows[i]["path"] for i in idx[ok].tolist()]
            if b % 200 == 0:
                print(f"часть {a.shard}: {len(names)} из {len(rows)}, {len(names) * k / (time.time() - t0):.1f} картинок/с", flush=True)
    tmp = out / f"shard_{a.shard:02d}.tmp.npy"
    np.save(tmp, np.concatenate(res))
    (out / f"shard_{a.shard:02d}.json").write_text(json.dumps(names, ensure_ascii=False))
    tmp.rename(path)
    print(f"часть {a.shard}: готово {len(names)} кадров, битых {bad}, {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
