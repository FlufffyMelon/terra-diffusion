#!/usr/bin/env python
"""
Чистит датасет после build_manifest.py. Кадры Терры не трогает: они проверены глазами.

Сначала считает признаки каждого кадра и сохраняет в filter_features.npz, потом применяет пороги,
поэтому пороги можно менять без пересчета (--reuse).

  CLIP по текстовым описаниям : рисунки, скриншоты, коллажи, игрушки, клетки, мертвые кошки, кадры без кошки
  бокс кошки                  : размытые, слишком темные и пересвеченные
  похожесть CLIP              : дубликаты и серии почти одинаковых кадров, из них остается самый большой
  лимит на особь              : Cat Individuals, не больше --max-per-cat кадров на кошку
  лимит на наблюдение         : iNaturalist, не больше --max-per-obs фото

В датасете: manifest.jsonl становится чистым, полный лежит в manifest_full.jsonl, отброшенное в rejected.jsonl
с причиной, ссылки на отброшенные кадры удаляются, в review/ листы случайных отказов по каждой причине.
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
# первая группа описывает нужные кадры, остальные причины отказа
PROMPTS = {
    "good": ["a photo of a cat", "a photo of a cat at home", "a photo of a cat outdoors", "a close-up photo of a cat"],
    "рисунок": ["a drawing of a cat", "a cartoon of a cat", "a painting of a cat", "a digital illustration of a cat"],
    "экран и текст": ["a screenshot of a phone", "a photo of a computer screen", "a poster with large text", "an image with a big watermark"],
    "коллаж": ["a collage of several photos", "a grid of many images"],
    "игрушка": ["a stuffed toy cat", "a cat statue", "a cat figurine", "a cat costume mascot"],
    "клетка": ["a cat inside a metal wire cage", "a cat in an animal shelter cage", "a pink wire pet crate", "a cat behind bars"],
    "мертвая": ["a dead cat", "a dead animal on the road"],
    "нет кошки": ["a photo of a dog", "a photo of a person", "a photo of an empty room", "a photo of a plant"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--reuse", action="store_true", help="взять признаки из filter_features.npz")
    p.add_argument("--dry-run", action="store_true", help="только посчитать и нарисовать листы, манифест не менять")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--thr", type=float, default=0.5, help="порог вероятности причины отказа")
    p.add_argument("--thr-cage", type=float, default=0.3, help="порог для клеток: они хуже всего для фона")
    p.add_argument("--min-sharp", type=float, default=25.0, help="дисперсия лапласиана бокса 224x224")
    p.add_argument("--min-bright", type=float, default=25.0)
    p.add_argument("--max-bright", type=float, default=235.0)
    p.add_argument("--dup-cos", type=float, default=0.97, help="сходство CLIP, выше которого кадры считаем дубликатами")
    p.add_argument("--max-per-cat", type=int, default=8)
    p.add_argument("--max-per-obs", type=int, default=2)
    return p.parse_args()


class Frames(Dataset):
    """Кадр -> вход CLIP 224x224 и резкость и яркость бокса кошки. Битый файл: ok = False."""

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
    tok = CLIPTokenizer.from_pretrained(CLIP, local_files_only=True)
    texts = [t for g in PROMPTS.values() for t in g]
    owner = np.array([k for k, g in enumerate(PROMPTS.values()) for _ in g])
    txt = model.get_text_features(**tok(texts, padding=True, return_tensors="pt").to("cuda"))
    txt = F.normalize(getattr(txt, "pooler_output", txt), dim=-1)
    embs, probs, sharp, bright, ok = [], [], [], [], []
    loader = DataLoader(Frames(rows, root), batch_size=a.batch, num_workers=a.workers, prefetch_factor=4)
    t0, n = time.time(), 0
    for b, (x, s, br, good) in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.get_image_features(pixel_values=x.cuda(non_blocking=True))
        e = F.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
        p = (100 * e @ txt.float().T).softmax(-1).cpu().numpy()
        probs.append(np.stack([p[:, owner == k].sum(1) for k in range(len(PROMPTS))], 1))
        embs.append(e.half().cpu().numpy())
        sharp.append(s.numpy()); bright.append(br.numpy()); ok.append(good.numpy())
        n += len(x)
        if b % 20 == 0:
            print(f"признаки: {n} из {len(rows)}, {n / (time.time() - t0):.0f} кадр/с", flush=True)
    return dict(emb=np.concatenate(embs), probs=np.concatenate(probs), sharp=np.concatenate(sharp),
                bright=np.concatenate(bright), ok=np.concatenate(ok))


def duplicates(emb, order, cos):
    """Номера кадров, у которых среди уже оставленных есть почти такой же. order: сначала более ценные кадры."""
    e = torch.from_numpy(emb[order]).float().cuda()
    keep = torch.ones(len(e), dtype=torch.bool, device="cuda")
    for s in range(0, len(e), 4096):
        q = e[s:s + 4096]
        m = (q @ e[:s + len(q)].T) >= cos                       # сравниваем только с теми, кто раньше в порядке
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
    names = list(PROMPTS)
    reasons = {}
    for i, r in enumerate(rows):
        if r["source"] == "terra":
            continue
        if not z["ok"][i]:
            reasons[i] = "битый файл"
            continue
        p = dict(zip(names, z["probs"][i]))
        if p["клетка"] >= a.thr_cage:
            reasons[i] = "клетка"
            continue
        bad = [n for n in names[1:] if n != "клетка" and p[n] >= a.thr]
        if bad:
            reasons[i] = max(bad, key=lambda n: p[n])
        elif z["sharp"][i] < a.min_sharp:
            reasons[i] = "размытый"
        elif z["bright"][i] < a.min_bright:
            reasons[i] = "темный"
        elif z["bright"][i] > a.max_bright:
            reasons[i] = "пересвеченный"
    left = np.array([i for i, r in enumerate(rows) if i not in reasons and r["source"] != "terra"])
    value = np.array([rows[i]["short"] + 1000 * rows[i]["fit_1x1"] for i in left])
    order = left[np.argsort(-value, kind="stable")]
    for i in duplicates(z["emb"], order, a.dup_cos):
        reasons[i] = "дубликат"
    counts = Counter()
    for i in order.tolist():                                   # сначала самые ценные кадры
        if i in reasons:
            continue
        r = rows[i]
        key, cap = (("cati", r["group"]), a.max_per_cat) if r["source"] == "cati" else (("inat", r.get("obs")), a.max_per_obs) if r["source"] == "inat" else (None, 0)
        if key:
            counts[key] += 1
            if counts[key] > cap:
                reasons[i] = "лимит на кошку"
    table = Counter(r["source"] for r in rows)
    rej = Counter((rows[i]["source"], w) for i, w in reasons.items())
    lines = ["", "## После фильтра", "", "| Источник | Было | Осталось | 1:1 |", "|---|---|---|---|"]
    for s in table:
        kept = [r for i, r in enumerate(rows) if r["source"] == s and i not in reasons]
        lines.append(f"| {s} | {table[s]} | {len(kept)} | {sum(r['fit_1x1'] for r in kept)} |")
    kept = [r for i, r in enumerate(rows) if i not in reasons]
    lines.append(f"| итого | {len(rows)} | {len(kept)} | {sum(r['fit_1x1'] for r in kept)} |")
    lines += ["", "Причины отказа:", ""] + [f"- {s}, {w}: {n}" for (s, w), n in sorted(rej.items())]
    print("\n".join(lines), flush=True)
    sheets(rows, reasons, root, root / "review")
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
    print("готово", flush=True)


if __name__ == "__main__":
    main()
