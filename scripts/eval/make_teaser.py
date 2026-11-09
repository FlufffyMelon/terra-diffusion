#!/usr/bin/env python
"""
Teaser image for the README: real photos of Terra on the top row, generated ones on the bottom row,
with a rotated label on the left of each row.

Real photos are cropped to a square around the cat's bounding box, so the cat stays inside the crop
as in training. Generated images are used as they are (512x512).

  .venv/bin/python scripts/eval/make_teaser.py --dataset ~/DiffusionLearn_data/pretrain_v1 \\
      --real terra/IMG_1.jpg terra/IMG_2.jpg ... --gen runs/teaser/gen/080.png ... --out assets/teaser.jpg
"""
import argparse
import json
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--real", nargs="+", required=True, help="paths from manifest.jsonl, e.g. terra/IMG_1234.jpg")
    p.add_argument("--gen", nargs="+", required=True, help="generated 512x512 images")
    p.add_argument("--out", required=True)
    p.add_argument("--side", type=int, default=384, help="side of one tile in pixels")
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--label", type=int, default=64, help="width of the label column")
    return p.parse_args()


def box_crop(path, box):
    """Square crop with the short side of the photo, centred on the bounding box and clamped to the photo."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    W, H = im.size
    s = min(W, H)
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    x = min(max(cx - s / 2, 0), W - s)
    y = min(max(cy - s / 2, 0), H - s)
    return im.crop((round(x), round(y), round(x) + s, round(y) + s))


def main():
    a = parse_args()
    root = Path(a.dataset)
    rows = {json.loads(l)["path"]: json.loads(l) for l in open(root / "manifest.jsonl")}
    real = [box_crop(root / p, rows[p]["box"]) for p in a.real]
    gen = [Image.open(p).convert("RGB") for p in a.gen]
    n = max(len(real), len(gen))
    S, G, L = a.side, a.gap, a.label
    canvas = Image.new("RGB", (L + n * (S + G) - G, 2 * S + G), (255, 255, 255))
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 30)
    for r, (label, ims) in enumerate((("original", real), ("generated", gen))):
        top = r * (S + G)
        for c, im in enumerate(ims):
            canvas.paste(im.resize((S, S), Image.LANCZOS), (L + c * (S + G), top))
        # the label is drawn horizontally on its own strip and rotated to read bottom to top
        strip = Image.new("RGB", (S, L), (255, 255, 255))
        d = ImageDraw.Draw(strip)
        w = d.textlength(label, font=font)
        d.text(((S - w) / 2, (L - 30) / 2 - 4), label, fill=(40, 40, 40), font=font)
        canvas.paste(strip.rotate(90, expand=True), (0, top))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out, quality=90)
    print(f"{a.out}: {canvas.size[0]}x{canvas.size[1]}")


if __name__ == "__main__":
    main()
