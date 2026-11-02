#!/usr/bin/env python
"""
Собирает датасет для претрейна из скачанных источников и детекций.

В <root>/<name>:
  manifest.jsonl : одна строка на кадр: source, group, path, box (доли кадра), W, H, short,
                   fit_1x1, fit_3x4, fit_4x3, license, author.
  terra/, inat/, oi/          : ссылки на фото,
  popcats/<порода>/           : деление источника сохранено,
  cati/<номер кошки>/         : то же.
  README.md                   : сколько кадров прошло и почему отброшены остальные.

Условия: обе модели нашли кошку с уверенностью >= --score и боксы совпали с IoU >= --iou, итоговый бокс это объединение.
Open Images: бокс размечен человеком, кадр с одной кошкой. Площадь бокса >= --min-area, кошка одна, снимок цветной.
iNaturalist: наблюдения с пометкой dead отброшены. Терра берется целиком, боксы проверены глазами.
Кадр попадает в манифест, если годится хотя бы для одного формата, кроп не растягивает фото.
"""
import argparse
import json
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

FORMATS = {"fit_1x1": (512, 512), "fit_3x4": (512, 384), "fit_4x3": (384, 512)}   # высота и ширина выхода
LICENSES = {"oi": "CC BY 2.0", "cati": "CC BY 4.0", "popcats": "не указана"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(Path.home() / "DiffusionLearn_data"))
    p.add_argument("--project", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--name", type=str, default="pretrain_v1")
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--min-area", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--limit", type=int, default=0, help="для проверки: кадров на источник")
    return p.parse_args()


def load(root, model, source):
    res = {}
    with open(root / "detections" / f"{model}_{source}.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # задачу могли остановить посреди записи последней строки
                continue
            res[r["f"]] = r
    return res


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def cats(rec, score):
    """Боксы кошек выше порога по убыванию уверенности, почти совпадающие боксы считаем одной кошкой."""
    out = []
    for b in sorted((b for b in rec["b"] if b[0] >= score), key=lambda b: -b[0]):
        if all(iou(b[1:], c) < 0.5 for c in out):
            out.append(b[1:])
    return out


def fits(box, W, H):
    """Для каждого формата: помещается ли бокс в кроп этого формата без растягивания фото."""
    bw, bh = (box[2] - box[0]) * W, (box[3] - box[1]) * H
    res = {}
    for name, (oh, ow) in FORMATS.items():
        k = oh / ow
        res[name] = bool(max(bh, bw * k, oh) <= min(H, W * k))
    return res


def is_gray(path):
    """Чёрно-белый снимок: каналы почти не отличаются на уменьшенной копии."""
    try:
        im = Image.open(path)
        im.draft("RGB", (128, 128))
        x = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
    except Exception:
        return True
    return float(np.abs(x[..., 0] - x[..., 1]).mean() + np.abs(x[..., 1] - x[..., 2]).mean()) < 3.0


def frames(a, root, source, folder, drop):
    """Кадры источника, прошедшие детекцию, с причинами отказа в счетчике drop."""
    Y, D = load(root, "yolo26x", source), load(root, "rtdetr-x", source)
    human = json.loads((root / "oi/boxes.json").read_text()) if source == "oi" else {}
    meta = {}
    dead = set()
    if source == "inat":
        dead = set(json.loads((root / "inat/dead_obs.json").read_text()))
        with open(root / "inat/photos.jsonl") as f:
            meta = {str(r["id"]): r for r in map(json.loads, f)}
    keys = sorted(k for k in Y if k in D)
    for k in keys[:a.limit or None]:
        y, d = Y[k], D[k]
        W, H = y["W"], y["H"]
        if W == 0:
            drop["битый файл"] += 1
            continue
        cy, cd = cats(y, a.score), cats(d, a.score)
        if not cy or not cd:
            drop["кошка не найдена обеими моделями"] += 1
            continue
        if len(cy) > 1 or len(cd) > 1:
            drop["несколько кошек"] += 1
            continue
        if iou(cy[0], cd[0]) < a.iou:
            drop["модели не согласны"] += 1
            continue
        box = [min(cy[0][0], cd[0][0]), min(cy[0][1], cd[0][1]), max(cy[0][2], cd[0][2]), max(cy[0][3], cd[0][3])]
        license_, author, stem = LICENSES.get(source), None, Path(k).stem
        if source == "oi":
            hb = human.get(stem, {}).get("boxes", [])
            if len(hb) != 1:
                drop["несколько кошек"] += 1
                continue
            box = [hb[0]["x1"], hb[0]["y1"], hb[0]["x2"], hb[0]["y2"]]
        if source == "inat":
            m = meta.get(stem)
            if m is None or m["obs"] in dead:
                drop["наблюдение dead или нет лицензии"] += 1
                continue
            license_, author = m["license"], m["attribution"]
        if (box[2] - box[0]) * (box[3] - box[1]) < a.min_area:
            drop["кошка меньше 10% кадра"] += 1
            continue
        f = fits(box, W, H)
        if not any(f.values()):
            drop["не годится ни в один формат"] += 1
            continue
        parts = Path(k).parts
        group = {"popcats": parts[1] if len(parts) > 2 else None, "cati": parts[1] if len(parts) > 2 else None}.get(source)
        rel = Path(source) / (group or "") / Path(k).name
        yield dict(source=source, group=group, path=str(rel), box=[round(v, 4) for v in box], W=W, H=H, short=min(W, H), **f,
                   license=license_, author=author), folder / k


def terra_frames(a):
    boxes = json.loads((Path(a.project) / "dataset_unique_labeled/boxes.json").read_text())
    src = Path(a.project) / "dataset_unique"
    for n, r in sorted(boxes.items()):
        W, H, b = r["width"], r["height"], r["box"]
        box = [round(b["x1"] / W, 4), round(b["y1"] / H, 4), round(b["x2"] / W, 4), round(b["y2"] / H, 4)]
        f = fits(box, W, H)
        if any(f.values()):
            yield dict(source="terra", group=None, path=f"terra/{n}", box=box, W=W, H=H, short=min(W, H), **f,
                       license="личные фото", author=None), src / n


def main():
    a = parse_args()
    t0 = time.time()
    root = Path(a.root)
    out = root / a.name
    if out.exists():
        shutil.rmtree(out)    # внутри только манифест, отчет и ссылки
    out.mkdir(parents=True)
    drop = Counter()
    items = list(terra_frames(a))
    sources = {"inat": root / "inat/images", "oi": root / "oi/images", "popcats": root / "probe/popcats", "cati": root / "probe/cati"}
    report = {}
    for s, folder in sources.items():
        d = Counter()
        got = list(frames(a, root, s, folder, d))
        report[s] = d
        items += got
        print(f"{s}: прошло детекцию {len(got)}, отброшено {sum(d.values())}: {dict(d)}, {time.time() - t0:.0f} с", flush=True)
    with Pool(a.workers) as pool:
        gray = pool.map(is_gray, [p for r, p in items], chunksize=64)
    keep = []
    for (r, p), g in zip(items, gray):
        if g and r["source"] != "terra":
            report[r["source"]]["чёрно-белый или не читается"] += 1
        else:
            keep.append((r, p))
    with open(out / "manifest.jsonl", "w") as f:
        for r, p in keep:
            dst = out / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(p)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# {a.name}", "", "| Источник | Всего | 1:1 | 3:4 | 4:3 |", "|---|---|---|---|---|"]
    for s in ["terra", *sources, None]:
        rows = [r for r, _ in keep if s is None or r["source"] == s]
        lines.append(f"| {s or 'итого'} | {len(rows)} | {sum(r['fit_1x1'] for r in rows)} | {sum(r['fit_3x4'] for r in rows)} | {sum(r['fit_4x3'] for r in rows)} |")
    lines += ["", "Отброшено:", ""]
    lines += [f"- {s}: {dict(d)}" for s, d in report.items()]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"готово за {time.time() - t0:.0f} с: {out}", flush=True)


if __name__ == "__main__":
    main()
