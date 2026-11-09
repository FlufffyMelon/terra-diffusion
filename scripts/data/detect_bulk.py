#!/usr/bin/env python
"""
Runs the detector over all downloaded cat photos in large batches. One model per GPU.

The run can be interrupted and repeated: finished files are skipped. With --watch, after processing everything
it waits for new files, so the job can be started while the download is still running.

<root>/detections/<model>_<source>.jsonl has one line per file:
  f : path relative to the source folder,  W, H : photo size after EXIF rotation,
  b : all cat boxes [confidence, x1, y1, x2, y2], coordinates as fractions of the frame.
No confidence filter above --conf: thresholds and model agreement are computed later, without rerunning.

Example:
  sbatch -p AMD7V12-RTX4500-CCP --gpus=1 -c 16 --mem=48G -t 08:00:00 -o runs/detect_%j.log \
      --wrap ".venv/bin/python -I scripts/data/detect_bulk.py --model yolo26x --watch 60"
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from ultralytics import RTDETR, YOLO

IMG_EXT = {".jpg", ".jpeg", ".png"}
CAT = 15            # COCO class
PAD = 114 / 255     # gray padding, as in ultralytics letterbox
MODELS = {"yolo26x": (YOLO, "yolo26x.pt"), "rtdetr-x": (RTDETR, "rtdetr-x.pt")}
SOURCES = {"popcats": "probe/popcats", "cati": "probe/cati", "oi": "oi/images", "inat": "inat/images"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--root", type=str, default=str(Path.home() / "DiffusionLearn_data"))
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0, help="for testing: how many files to process")
    p.add_argument("--watch", type=int, default=0, help="how many minutes to wait for new files after processing everything")
    return p.parse_args()


class Images(Dataset):
    """Photo -> size x size square with gray padding. Returns the image, its dimensions and the padding. Broken file: W = 0."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        s = self.size
        try:
            im = Image.open(self.paths[i])
            w0, h0 = im.size
            im.draft("RGB", (2 * s, 2 * s))  # fast JPEG decoding directly at reduced size
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = (w0, h0) if (im.width >= im.height) == (w0 >= h0) else (h0, w0)
            k = s / max(im.size)
            nw, nh = max(round(im.width * k), 1), max(round(im.height * k), 1)
            x = TF.pil_to_tensor(im.resize((nw, nh), Image.BILINEAR))
            px, py = (s - nw) // 2, (s - nh) // 2
            out = torch.full((3, s, s), round(PAD * 255), dtype=torch.uint8)
            out[:, py:py + nh, px:px + nw] = x
            return out, i, W, H, nw, nh, px, py
        except Exception:
            return torch.zeros((3, s, s), dtype=torch.uint8), i, 0, 0, 1, 1, 0, 0


def pending(src_dir, done):
    """Files in the folder that are not in the results yet. Unfinished downloads (.tmp) are left alone."""
    if not src_dir.exists():
        return []
    return sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()
                  and str(p.relative_to(src_dir)) not in done)


def run_source(model, a, src_dir, paths, out_path):
    loader = DataLoader(Images(paths, a.size), batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                        prefetch_factor=4, persistent_workers=False)
    t0, n = time.time(), 0
    with open(out_path, "a") as f:
        for b, (x, idx, W, H, nw, nh, px, py) in enumerate(loader):
            x = x.to(0, non_blocking=True).float() / 255
            res = model.predict(x, conf=a.conf, classes=[CAT], imgsz=a.size, half=True, device=0, verbose=False)
            for j, r in enumerate(res):
                boxes = []
                for s, q in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    gx, gy, kx, ky = px[j].item(), py[j].item(), nw[j].item(), nh[j].item()
                    c = [(q[0] - gx) / kx, (q[1] - gy) / ky, (q[2] - gx) / kx, (q[3] - gy) / ky]
                    boxes.append([round(s, 4)] + [round(min(max(v, 0.0), 1.0), 4) for v in c])
                f.write(json.dumps(dict(f=str(paths[idx[j]].relative_to(src_dir)), W=W[j].item(), H=H[j].item(), b=boxes)) + "\n")
            f.flush()
            n += len(res)
            if b % 20 == 0:
                print(f"  {out_path.stem}: {n} of {len(paths)}, {n / (time.time() - t0):.1f} frames/s", flush=True)
    print(f"  {out_path.stem}: done {n}, {n / (time.time() - t0):.1f} frames/s", flush=True)


def main():
    a = parse_args()
    root = Path(a.root)
    out = root / "detections"
    out.mkdir(exist_ok=True)
    cls, weights = MODELS[a.model]
    model = cls(str(root / "weights" / weights))
    idle = 0
    while True:
        worked = False
        for name in a.sources:
            src_dir = root / SOURCES[name]
            out_path = out / f"{a.model}_{name}.jsonl"
            done = {json.loads(l)["f"] for l in open(out_path)} if out_path.exists() else set()
            paths = pending(src_dir, done)
            paths = paths[:a.limit] if a.limit else paths
            if paths:
                print(f"{name}: new files {len(paths)}, done earlier {len(done)}", flush=True)
                run_source(model, a, src_dir, paths, out_path)
                worked = True
        if a.limit or (not worked and idle >= a.watch):
            break
        idle = 0 if worked else idle + 1
        if not worked:
            time.sleep(60)
    print("done", flush=True)


if __name__ == "__main__":
    main()
