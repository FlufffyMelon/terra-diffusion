#!/usr/bin/env python
"""
Cleans the dataset after build_manifest.py. Terra frames are left alone: they were checked by eye.

First computes features for each frame and saves them to filter_features.npz, then applies the thresholds.
With --reuse the images are not recomputed: probabilities for the text prompts are recomputed from the saved CLIP embeddings,
so both the thresholds and the prompts themselves in PROMPTS can be changed.

  CLIP text prompts          : drawings, screenshots, collages, toys, cages, dead cats, dead prey, frames without a cat
  cat box                    : blurry, too dark and overexposed
  CLIP similarity           : duplicates and series of nearly identical frames, the largest one is kept
  per-individual limit       : Cat Individuals, at most --max-per-cat frames per cat
  per-observation limit      : iNaturalist, at most --max-per-obs photos

In the dataset: manifest.jsonl becomes the clean one, the full one is in manifest_full.jsonl, the dropped ones in rejected.jsonl
with the reason, links to dropped frames are removed, review/ holds sheets of random rejections for each reason.
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, CLIPTokenizer

CLIP = "openai/clip-vit-large-patch14"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
# the first group describes the wanted frames, the rest are rejection reasons
PROMPTS = {
    "good": ["a photo of a cat", "a photo of a cat at home", "a photo of a cat outdoors", "a close-up photo of a cat"],
    "drawing": ["a drawing of a cat", "a cartoon of a cat", "a painting of a cat", "a digital illustration of a cat"],
    "screen_text": ["a screenshot of a phone", "a photo of a computer screen", "a poster with large text", "an image with a big watermark"],
    "collage": ["a collage of several photos", "a grid of many images"],
    "toy": ["a stuffed toy cat", "a cat statue", "a cat figurine", "a cat costume mascot"],
    "cage": ["a cat inside a metal wire cage", "a cat in an animal shelter cage", "a pink wire pet crate", "a cat behind bars"],
    "dead_cat": ["a dead cat", "a dead animal on the road"],
    "prey": ["a cat with a dead mouse", "a dead mouse on the floor", "a dead bird", "a cat holding prey in its mouth",
             "a cat eating a mouse", "a dead rat"],
    "no_cat": ["a photo of a dog", "a photo of a person", "a photo of an empty room", "a photo of a plant"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--reuse", action="store_true", help="take features from filter_features.npz")
    p.add_argument("--dry-run", action="store_true", help="only compute and draw the sheets, do not change the manifest")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--thr", type=float, default=0.5, help="probability threshold for a rejection reason")
    p.add_argument("--thr-cage", type=float, default=0.3, help="threshold for cages: they are the worst for the background")
    p.add_argument("--thr-prey", type=float, default=0.3, help="threshold for dead prey: such frames are unacceptable")
    p.add_argument("--top", default="", help="reason to draw review/top_<reason>.jpg for: the most likely frames")
    p.add_argument("--min-sharp", type=float, default=25.0, help="Laplacian variance of the 224x224 box")
    p.add_argument("--min-bright", type=float, default=25.0)
    p.add_argument("--max-bright", type=float, default=235.0)
    p.add_argument("--dup-cos", type=float, default=0.97, help="CLIP similarity above which frames are considered duplicates")
    p.add_argument("--max-per-cat", type=int, default=8)
    p.add_argument("--max-per-obs", type=int, default=2)
    return p.parse_args()


class Frames(Dataset):
    """Frame -> 224x224 CLIP input plus sharpness and brightness of the cat box. Broken file: ok = False."""

    def __init__(self, rows, root):
        self.rows, self.root = rows, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = Image.open(self.root / r["path"])
            im.draft("RGB", (512, 512))
            im = ImageOps.exif_transpose(im).convert("RGB")
            b = r["box"]
            W, H = im.size
            crop = im.crop((int(b[0] * W), int(b[1] * H), max(int(b[2] * W), int(b[0] * W) + 1), max(int(b[3] * H), int(b[1] * H) + 1)))
            g = np.asarray(crop.convert("L").resize((224, 224), Image.BILINEAR), dtype=np.float32)
            lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
            x = TF.center_crop(TF.resize(TF.pil_to_tensor(im), 224, interpolation=InterpolationMode.BICUBIC, antialias=True), [224, 224])
            x = TF.normalize(x.float() / 255, MEAN, STD)
            return x, float(lap.var()), float(g.mean()), True
        except Exception:
            return torch.zeros(3, 224, 224), 0.0, 0.0, False


@torch.inference_mode()
def compute(a, rows, root):
    model = CLIPModel.from_pretrained(CLIP, local_files_only=True).cuda().eval()
    embs, sharp, bright, ok = [], [], [], []
    loader = DataLoader(Frames(rows, root), batch_size=a.batch, num_workers=a.workers, prefetch_factor=4)
    t0, n = time.time(), 0
    for b, (x, s, br, good) in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.get_image_features(pixel_values=x.cuda(non_blocking=True))
        e = F.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
        embs.append(e.half().cpu().numpy())
        sharp.append(s.numpy()); bright.append(br.numpy()); ok.append(good.numpy())
        n += len(x)
        if b % 20 == 0:
            print(f"features: {n} of {len(rows)}, {n / (time.time() - t0):.0f} frames/s", flush=True)
    return dict(emb=np.concatenate(embs), sharp=np.concatenate(sharp), bright=np.concatenate(bright), ok=np.concatenate(ok))


@torch.inference_mode()
def text_probs(emb):
    """Probability of each PROMPTS group for each frame from CLIP embeddings: softmax over all prompts, summed over the group."""
    model = CLIPModel.from_pretrained(CLIP, local_files_only=True).cuda().eval()
    tok = CLIPTokenizer.from_pretrained(CLIP, local_files_only=True)
    texts = [t for g in PROMPTS.values() for t in g]
    owner = np.array([k for k, g in enumerate(PROMPTS.values()) for _ in g])
    txt = model.get_text_features(**tok(texts, padding=True, return_tensors="pt").to("cuda"))
    txt = F.normalize(getattr(txt, "pooler_output", txt).float(), dim=-1)
    out = []
    for s in range(0, len(emb), 8192):
        p = (100 * torch.from_numpy(emb[s:s + 8192]).float().cuda() @ txt.T).softmax(-1).cpu().numpy()
        out.append(np.stack([p[:, owner == k].sum(1) for k in range(len(PROMPTS))], 1))
    return np.concatenate(out)


def top_sheet(rows, p, root, path, n=96, T=220, C=12):
    """Frames with the highest probability p, caption: probability and source. For picking a threshold by eye."""
    font = ImageFont.load_default(size=13)
    idx = np.argsort(-p)[:n]
    S = Image.new("RGB", (C * T, ((len(idx) + C - 1) // C) * T), (25, 25, 25))
    for j, i in enumerate(idx):
        try:
            im = ImageOps.pad(ImageOps.exif_transpose(Image.open(root / rows[i]["path"])).convert("RGB"), (T - 4, T - 4), color=(25, 25, 25))
        except Exception:
            continue
        ImageDraw.Draw(im).text((3, 3), f"{p[i]:.2f} {rows[i]['source']}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        S.paste(im, ((j % C) * T + 2, (j // C) * T + 2))
    S.save(path, quality=85)


def duplicates(emb, order, cos):
    """Indices of frames that have a nearly identical one among those already kept. order: more valuable frames first."""
    e = torch.from_numpy(emb[order]).float().cuda()
    keep = torch.ones(len(e), dtype=torch.bool, device="cuda")
    for s in range(0, len(e), 4096):
        q = e[s:s + 4096]
        m = (q @ e[:s + len(q)].T) >= cos                       # compare only with those earlier in the order
        m = m & keep[:s + len(q)][None, :]
        for j in range(len(q)):
            m[j, s + j:] = False
        keep[s:s + len(q)] = ~m.any(1)
    return set(order[(~keep).cpu().numpy()].tolist())


def sheets(rows, reasons, root, out, n=48, T=220, C=8):
    font = ImageFont.load_default(size=12)
    out.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for i, why in reasons.items():
        groups[why].append(i)
    random.seed(0)
    for why, idx in groups.items():
        pick = random.sample(idx, min(n, len(idx)))
        S = Image.new("RGB", (C * T, ((len(pick) + C - 1) // C) * T), (25, 25, 25))
        for j, i in enumerate(pick):
            try:
                im = ImageOps.pad(ImageOps.exif_transpose(Image.open(root / rows[i]["path"])).convert("RGB"), (T - 4, T - 4), color=(25, 25, 25))
            except Exception:
                continue
            ImageDraw.Draw(im).text((3, 3), rows[i]["source"], fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            S.paste(im, ((j % C) * T + 2, (j // C) * T + 2))
        S.save(out / f"{why.replace(' ', '_')}.jpg", quality=85)


def main():
    a = parse_args()
    root = Path(a.dataset)
    full = root / "manifest_full.jsonl"
    src = full if full.exists() else root / "manifest.jsonl"
    with open(src) as f:
        rows = [json.loads(l) for l in f]
    feat_path = root / "filter_features.npz"
    if a.reuse:
        z = dict(np.load(feat_path))
    else:
        z = compute(a, rows, root)
        np.savez(feat_path, **z)
    z["probs"] = text_probs(z["emb"])
    names = list(PROMPTS)
    reasons = {}
    for i, r in enumerate(rows):
        if r["source"] == "terra":
            continue
        if not z["ok"][i]:
            reasons[i] = "broken_file"
            continue
        p = dict(zip(names, z["probs"][i]))
        if p["prey"] >= a.thr_prey:
            reasons[i] = "prey"
            continue
        if p["cage"] >= a.thr_cage:
            reasons[i] = "cage"
            continue
        bad = [n for n in names[1:] if n not in ("cage", "prey") and p[n] >= a.thr]
        if bad:
            reasons[i] = max(bad, key=lambda n: p[n])
        elif z["sharp"][i] < a.min_sharp:
            reasons[i] = "blurry"
        elif z["bright"][i] < a.min_bright:
            reasons[i] = "dark"
        elif z["bright"][i] > a.max_bright:
            reasons[i] = "overexposed"
    # dropped by a previous filter run (the link is already gone) and the rules now let it through: the decision is not revisited,
    # otherwise borderline frames come back without a file and without latents
    for i, r in enumerate(rows):
        if i not in reasons and r["source"] != "terra" and not (root / r["path"]).exists():
            reasons[i] = "rejected_earlier"
    left = np.array([i for i, r in enumerate(rows) if i not in reasons and r["source"] != "terra"])
    value = np.array([rows[i]["short"] + 1000 * rows[i]["fit_1x1"] for i in left])
    order = left[np.argsort(-value, kind="stable")]
    for i in duplicates(z["emb"], order, a.dup_cos):
        reasons[i] = "duplicate"
    counts = Counter()
    for i in order.tolist():                                   # most valuable frames first
        if i in reasons:
            continue
        r = rows[i]
        key, cap = (("cati", r["group"]), a.max_per_cat) if r["source"] == "cati" else (("inat", r.get("obs")), a.max_per_obs) if r["source"] == "inat" else (None, 0)
        if key:
            counts[key] += 1
            if counts[key] > cap:
                reasons[i] = "per_cat_limit"
    table = Counter(r["source"] for r in rows)
    rej = Counter((rows[i]["source"], w) for i, w in reasons.items())
    lines = ["", "## After filter", "", "| Source | Before | Kept | 1:1 |", "|---|---|---|---|"]
    for s in table:
        kept = [r for i, r in enumerate(rows) if r["source"] == s and i not in reasons]
        lines.append(f"| {s} | {table[s]} | {len(kept)} | {sum(r['fit_1x1'] for r in kept)} |")
    kept = [r for i, r in enumerate(rows) if i not in reasons]
    lines.append(f"| total | {len(rows)} | {len(kept)} | {sum(r['fit_1x1'] for r in kept)} |")
    lines += ["", "Rejection reasons:", ""] + [f"- {s}, {w}: {n}" for (s, w), n in sorted(rej.items())]
    print("\n".join(lines), flush=True)
    sheets(rows, reasons, root, root / "review")
    if a.top:
        p = z["probs"][:, names.index(a.top)].copy()
        # Terra is left alone; frames dropped by a previous filter run no longer have links
        p[[r["source"] == "terra" or not (root / r["path"]).exists() for r in rows]] = -1
        top_sheet(rows, p, root, root / "review" / f"top_{a.top.replace(' ', '_')}.jpg")
        for thr in (0.1, 0.2, 0.3, 0.5):
            print(f"{a.top}: p >= {thr}: {(p >= thr).sum()} frames", flush=True)
    if a.dry_run:
        return
    if not full.exists():
        (root / "manifest.jsonl").rename(full)
    with open(root / "manifest.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(root / "rejected.jsonl", "w") as f:
        for i, w in reasons.items():
            f.write(json.dumps(dict(path=rows[i]["path"], why=w), ensure_ascii=False) + "\n")
            (root / rows[i]["path"]).unlink(missing_ok=True)
    with open(root / "README.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
