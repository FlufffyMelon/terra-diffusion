#!/usr/bin/env python
"""
Находит кошку на каждом фото детектором с весами COCO и рисует бокс для проверки глазами.

На фото оставляем один бокс: класс cat с наибольшей уверенностью, не ниже --score.
В папке <data-dir>_labeled появляются:
  boxes.json : для каждого фото размеры и бокс, у фото без кошки box = null. Координаты в пикселях
               исходного фото после поворота по EXIF, как в load_images из train.py.
  фото       : с теми же именами, с нарисованным боксом, длинная сторона не больше --view-size.
  no_cat/    : фото, где кошка не найдена, с --draw-top самыми уверенными предсказаниями любых классов.
               Номер на рисунке совпадает с id в no_cat/predictions.json, там лежат все предсказания,
               чтобы выбрать бокс глазами.

Пример:
  sbatch -p AMD7V12-RTX4500-CCP --gpus=1 -c 8 --mem=32G -t 00:30:00 -o runs/detect_%j.log \
      --wrap ".venv/bin/python detect_cats.py --data-dir dataset_unique"
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from train import IMG_EXT

GREEN, RED = (0, 230, 0), (255, 0, 0)
PALETTE = [(255, 220, 0), (0, 200, 255), (255, 90, 200), (255, 140, 0), (120, 255, 120), (200, 160, 255)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="dataset_unique")
    p.add_argument("--score", type=float, default=0.3, help="минимальная уверенность, чтобы считать кошку найденной")
    p.add_argument("--min-score", type=float, default=0.01, help="ниже этой уверенности детектор ничего не выдает")
    p.add_argument("--view-size", type=int, default=1024, help="длинная сторона картинок с боксами")
    p.add_argument("--draw-top", type=int, default=12, help="сколько предсказаний рисовать на фото без кошки")
    return p.parse_args()


def predict(model, img, names):
    """Все предсказания для фото, по убыванию уверенности. id = место в этом списке."""
    out = model([TF.to_dtype(TF.pil_to_tensor(img), torch.float32, scale=True).to(next(model.parameters()).device)])[0]
    preds = [dict(label=names[lab], score=round(s, 4), x1=round(box[0], 1), y1=round(box[1], 1),
                  x2=round(box[2], 1), y2=round(box[3], 1))
             for box, lab, s in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())]
    preds.sort(key=lambda p: -p["score"])
    return [dict(id=i, **p) for i, p in enumerate(preds)]


def draw(img, boxes, view_size, numbered=False):
    """Копия фото с боксами. numbered: разные цвета и номера предсказаний, красная рамка по краю кадра."""
    k = min(1.0, view_size / max(img.size))
    view = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    d = ImageDraw.Draw(view)
    font = ImageFont.load_default(size=max(14, view_size // 40))
    lw = max(3, view_size // 250)
    if numbered:
        d.rectangle([0, 0, view.width - 1, view.height - 1], outline=RED, width=3 * lw)
    for b in reversed(boxes):
        color = PALETTE[b["id"] % len(PALETTE)] if numbered else GREEN
        tag = f"#{b['id']} " if numbered else ""
        xy = [b["x1"] * k, b["y1"] * k, b["x2"] * k, b["y2"] * k]
        d.rectangle(xy, outline=color, width=lw)
        score = "" if b["score"] is None else f" {b['score']:.2f}"  # у боксов, собранных вручную, уверенности нет
        d.text((xy[0] + lw + 4, xy[1] + lw + 2), f"{tag}{b['label']}{score}", fill=color, font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    return view


def main():
    a = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Path(a.data_dir)
    out = src.with_name(src.name + "_labeled")
    (out / "no_cat").mkdir(parents=True, exist_ok=True)

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    names = weights.meta["categories"]
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=a.min_score).to(dev).eval()

    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
    print(f"фото {len(files)}, устройство {dev}", flush=True)
    t0 = time.time()
    boxes, no_cat = {}, {}
    with torch.inference_mode():
        for i, p in enumerate(files):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            preds = predict(model, img, names)
            cats = [b for b in preds if b["label"] == "cat" and b["score"] >= a.score]
            box = None
            if cats:
                box = {k: v for k, v in cats[0].items() if k != "id"}
                draw(img, [box], a.view_size).save(out / p.name, format="JPEG", quality=88)
            else:
                no_cat[p.name] = dict(width=img.width, height=img.height, predictions=preds)
                draw(img, preds[:a.draw_top], a.view_size, numbered=True).save(out / "no_cat" / p.name, format="JPEG", quality=88)
            boxes[p.name] = dict(width=img.width, height=img.height, box=box)
            if (i + 1) % 50 == 0:
                print(f"{i + 1} из {len(files)}, {time.time() - t0:.0f} с", flush=True)

    (out / "boxes.json").write_text(json.dumps(boxes, indent=1, ensure_ascii=False))
    (out / "no_cat" / "predictions.json").write_text(json.dumps(no_cat, indent=1, ensure_ascii=False))

    best = np.array([r["box"]["score"] for r in boxes.values() if r["box"]])
    print(f"кошка найдена на {len(best)} фото из {len(files)}, уверенность: медиана {np.median(best):.3f}, "
          f"мин {best.min():.3f}, ниже 0.9 на {int((best < 0.9).sum())} фото")
    print(f"без кошки {len(no_cat)}: " + ", ".join(f"{n} ({len(r['predictions'])} предсказаний)" for n, r in no_cat.items()))
    print(f"готово за {time.time() - t0:.0f} с, результаты в {out}")


if __name__ == "__main__":
    main()
