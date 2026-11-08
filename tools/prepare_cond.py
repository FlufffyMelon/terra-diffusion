#!/usr/bin/env python
"""
Условия для ldm_v4: отложенные фото Терры и кластеры по признакам DINOv2.
Кластеры получаются в основном по окрасу и породе (рыжие, смокинги, колорпойнт), а не по позе или фону,
поэтому в обучении это условие «тип кошки». Терра попадает в кластер длинношерстных колорпойнтов.

  terra_holdout.json : --holdout кадров Терры fit_1x1, которые не идут в обучение, на них проверяем похожесть и новизну.
  clusters.json      : номер кластера у каждого кадра fit_1x1, k-means по признакам DINOv2 ViT-L/14 (CLS, 224).
  dino_feats.npz     : сами признаки, цветные и черно-белые, для метрик и пересчета кластеров.
  review/clusters_*.jpg : по --per-sheet кластеров на лист, строка = кластер, случайные кадры из него.

Кластер сцены не должен означать Терру: тогда условие «Терра» спорило бы с условием «сцена».
Если считать k-means по всем кадрам, DINOv2 собирает почти все фото Терры в 1-3 своих кластера.
Поэтому центры считаются только по другим кошкам, а кадры Терры относятся к ближайшему центру.
Кластеры считаются по цветным и по черно-белым картинкам, берется вариант, где кадры Терры
разбросаны по кластерам сильнее (меньше доля самого частого кластера Терры).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import font_manager
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import FrameCrops, read_manifest

DINO = "facebook/dinov2-large"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--holdout", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-sheet", type=int, default=25)
    p.add_argument("--reuse", action="store_true", help="признаки из dino_feats.npz")
    return p.parse_args()


@torch.no_grad()
def features(rows, root, a):
    model = AutoModel.from_pretrained(DINO, local_files_only=True).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    color, gray, ok_all = [], [], []
    loader = DataLoader(FrameCrops(rows, root, size=224, crops=1, seed=0), batch_size=a.batch, num_workers=a.workers)
    for b, (x, _, ok) in enumerate(loader):
        x = (x[:, 0].cuda().float() + 1) / 2
        g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).expand(-1, 3, -1, -1)
        for img, acc in ((x, color), (g, gray)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(pixel_values=(img - mean) / std).pooler_output
            acc.append(F.normalize(f.float(), dim=-1).cpu())
        ok_all.append(ok)
        if b % 50 == 0:
            print(f"признаки: {(b + 1) * a.batch} из {len(rows)}", flush=True)
    return torch.cat(color), torch.cat(gray), torch.cat(ok_all)


def kmeans(x, k, iters=40, restarts=3, seed=0):
    """Сферический k-means (косинусная близость) с инициализацией k-means++. Возвращает метки и центры."""
    x = x.cuda()
    best = None
    for r in range(restarts):
        g = torch.Generator(device="cuda").manual_seed(seed + r)
        c = x[torch.randint(len(x), (1,), generator=g, device="cuda")]
        for _ in range(k - 1):
            d = (1 - (x @ c.T).max(1).values).clamp_min(0)
            c = torch.cat([c, x[torch.multinomial(d ** 2 + 1e-12, 1, generator=g)]])
        for _ in range(iters):
            lab = (x @ c.T).argmax(1)
            c = F.normalize(torch.zeros_like(c).index_add_(0, lab, x), dim=-1)
        lab = (x @ c.T).argmax(1)
        score = (x * c[lab]).sum(1).mean().item()
        if best is None or score > best[0]:
            best = (score, lab.cpu(), c.cpu())
    return best[1], best[2]


def sheets(rows, root, lab, k, path_fmt, per_sheet, n=10, T=150):
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 14)
    random.seed(0)
    for s in range(0, k, per_sheet):
        ids = list(range(s, min(s + per_sheet, k)))
        S = Image.new("RGB", (60 + n * T, len(ids) * T), (25, 25, 25))
        d = ImageDraw.Draw(S)
        for r, c in enumerate(ids):
            members = [i for i in range(len(rows)) if lab[i] == c]
            terra = sum(rows[i]["source"] == "terra" for i in members)
            d.text((4, r * T + 50), f"{c}\n{len(members)}\nT {terra}", fill=(255, 255, 255), font=font)
            for j, i in enumerate(random.sample(members, min(n, len(members)))):
                try:
                    im = ImageOps.fit(ImageOps.exif_transpose(Image.open(root / rows[i]["path"])).convert("RGB"), (T - 4, T - 4))
                except Exception:
                    continue
                S.paste(im, (60 + j * T + 2, r * T + 2))
        S.save(path_fmt.format(s // per_sheet), quality=85)


def main():
    a = parse_args()
    root = Path(a.dataset)
    rows = read_manifest(root / "manifest.jsonl")
    terra = [r["path"] for r in rows if r["source"] == "terra"]

    hold = root / "terra_holdout.json"
    if not hold.exists():
        hold.write_text(json.dumps(sorted(random.Random(0).sample(terra, a.holdout)), ensure_ascii=False, indent=1))
    print(f"Терра 1:1: {len(terra)} кадров, отложено {len(json.loads(hold.read_text()))}", flush=True)

    fpath = root / "dino_feats.npz"
    if a.reuse:
        z = np.load(fpath)
        color, gray, ok = torch.from_numpy(z["color"]), torch.from_numpy(z["gray"]), torch.from_numpy(z["ok"])
    else:
        color, gray, ok = features(rows, root, a)
        np.savez(fpath, color=color.numpy(), gray=gray.numpy(), ok=ok.numpy(), paths=np.array([r["path"] for r in rows]))
    is_terra = torch.tensor([r["source"] == "terra" for r in rows])

    best = None
    for name, f in (("color", color), ("gray", gray)):
        _, cent = kmeans(f[~is_terra], a.k)
        lab = (f @ cent.T).argmax(1)
        counts = torch.bincount(lab[is_terra], minlength=a.k)
        share = counts.max().item() / counts.sum().item()
        print(f"{name}: кадры Терры в {int((counts > 0).sum())} кластерах, самый частый {share:.0%}, "
              f"размеры кластеров от {torch.bincount(lab, minlength=a.k).min().item()} до {torch.bincount(lab, minlength=a.k).max().item()}", flush=True)
        if best is None or share < best[0]:
            best = (share, name, lab)
    _, name, lab = best
    (root / "clusters.json").write_text(json.dumps(dict(k=a.k, features=name, assign={r["path"]: int(l) for r, l in zip(rows, lab.tolist())}),
                                                   ensure_ascii=False))
    print(f"выбран вариант {name}, clusters.json записан", flush=True)
    (root / "review").mkdir(exist_ok=True)
    sheets(rows, root, lab.tolist(), a.k, str(root / "review" / "clusters_{}.jpg"), a.per_sheet)
    print("листы кластеров готовы", flush=True)


if __name__ == "__main__":
    main()
