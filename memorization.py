#!/usr/bin/env python
"""
Насколько модель копирует обучающие фото.

Для каждого чекпоинта генерируем N картинок из одного и того же шума и для каждой ищем
ближайшее обучающее фото среди всех вариантов кропа (масштабы, сдвиги, отражения), как при обучении.

  d1    = расстояние до ближайшего фото,
  d2    = расстояние до второго ближайшего (другого) фото,
  ratio = d1 / d2.

У копии d1 намного меньше, чем расстояние до любого другого фото, и ratio мал.
Размытая «средняя» картинка близка ко многим фото сразу, у нее ratio около 1,
поэтому тест не путает ее с копией (в отличие от одного d1).

Калибровка: то же самое для самих обучающих фото (центральный кроп) относительно всех остальных фото.
Это уровень, который дают настоящие, но разные снимки.

Качество: KID между признаками Inception-v3 у сэмплов и у настоящих фото (меньше = лучше).
Внимание, модель, которая просто копирует фото, получит очень хороший KID,
поэтому KID читаем только вместе с ratio.

Запуски, обученные с --boxes, проверяем с тем же --boxes: варианты кропа, калибровка и настоящие фото для KID
берутся так же, как при обучении, то есть бокс кошки всегда внутри кропа.

Пример:
  sbatch -p AMD7V12-RTX4500-CCP --gpus=1 -c 8 --mem=48G -t 02:00:00 -o runs/memorization_%j.log \
      --wrap ".venv/bin/python memorization.py --runs runs/cat64_eps runs/cat64_uniq \
              --data-dirs Terra dataset_unique --exclude-files exclude_eps.txt -"
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import model_from_checkpoint
import torchvision.transforms.v2.functional as TF
from train import (CatDataset, center_crops, crop_offsets, load_boxes, load_images, min_crop_side, sample, save_grid,
                   sde_from_cfg, to_unit)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="папки запусков")
    p.add_argument("--data-dirs", nargs="+", required=True, help="по одной папке с фото на каждый запуск")
    p.add_argument("--exclude-files", nargs="+", default=None,
                   help="по одному файлу исключений на каждый запуск, '-' если исключать нечего")
    p.add_argument("--n", type=int, default=512, help="сколько картинок генерировать на чекпоинт")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--method", default="heun", choices=["heun", "euler", "sde"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rho", type=float, default=2.0)
    p.add_argument("--scales", default="1.0,0.925,0.85", help="масштабы кропа, как при обучении")
    p.add_argument("--npos", type=int, default=4, help="сдвигов по каждой оси")
    p.add_argument("--boxes", default=None, help="boxes.json: кропы как при обучении с --boxes")
    p.add_argument("--nsides", type=int, default=5, help="сторон кропа на фото при --boxes")
    p.add_argument("--n-real", type=int, default=4, help="случайных кропов на фото для KID при --boxes")
    p.add_argument("--out", default="runs/memorization")
    p.add_argument("--no-kid", action="store_true", help="не считать KID (метрику качества)")
    return p.parse_args()


# ----------------------------------------------------------------------------- KID
def load_inception(dev):
    """Inception-v3 из torchvision, без последнего слоя: на выходе 2048 признаков на картинку.
    Веса скачиваются при первом вызове в ~/.cache/torch, на нодах без интернета скачай заранее."""
    import torchvision
    net = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    return net.to(dev).eval()


@torch.no_grad()
def inception_feats(net, x, batch=64):
    """x : (N, 3, S, S) в [-1, 1]  ->  (N, 2048) признаков в float64 на CPU."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(x), batch):
        b = (x[i:i + batch].float().clamp(-1, 1) + 1) / 2
        b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        out.append(net((b - mean) / std).double().cpu())
    return torch.cat(out)


def kid(f_real, f_gen, n_subsets=100, seed=0):
    """
    Kernel Inception Distance: несмещенная оценка MMD^2 с ядром k(a, b) = (a.b / d + 1)^3
    между признаками настоящих фото и сэмплов. 0 = распределения совпадают, меньше = лучше.
    Считаем по случайным подвыборкам одного размера, возвращаем среднее и разброс.
    """
    g = torch.Generator().manual_seed(seed)
    d = f_real.shape[1]
    m = min(len(f_real), len(f_gen))
    vals = []
    for _ in range(n_subsets):
        a = f_real[torch.randperm(len(f_real), generator=g)[:m]]
        b = f_gen[torch.randperm(len(f_gen), generator=g)[:m]]
        kaa = (a @ a.T / d + 1) ** 3
        kbb = (b @ b.T / d + 1) ** 3
        kab = (a @ b.T / d + 1) ** 3
        vals.append(((kaa.sum() - kaa.diagonal().sum()) / (m * (m - 1))
                     + (kbb.sum() - kbb.diagonal().sum()) / (m * (m - 1))
                     - 2 * kab.mean()).item())
    vals = np.array(vals)
    return vals.mean(), vals.std()


def read_exclude(path):
    ex = set()
    if path and path != "-":
        for line in Path(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                ex.add(line)
    return ex


def build_refs(imgs, size, scales, npos, boxes=None, nsides=5):
    """
    Все варианты кропа каждого фото: масштабы x сдвиги x отражение. Возвращает (M, 3, S, S) и номер фото.
    С boxes вместо масштабов nsides сторон от наименьшей допустимой до короткой стороны фото, бокс всегда внутри.
    """
    crops, owner = [], []
    for k, img in enumerate(imgs):
        _, H, W = img.shape
        short = min(H, W)
        if boxes is None:
            sides = [int(short * s) for s in scales]
        else:
            sides = sorted(set(np.linspace(min_crop_side(boxes[k], H, W, size), short, nsides).round().astype(int).tolist()))
        for side in sides:
            top_min, top_max, left_min, left_max = (0, H - side, 0, W - side) if boxes is None else crop_offsets(boxes[k], H, W, side)
            ys = sorted(set(np.linspace(top_min, top_max, npos).round().astype(int).tolist()))
            xs = sorted(set(np.linspace(left_min, left_max, npos).round().astype(int).tolist()))
            b = torch.stack([img[:, y:y + side, x:x + side] for y in ys for x in xs])        # uint8
            b = to_unit(TF.resize(b, [size, size], antialias=True))                          # как в обучении
            b = torch.cat([b, b.flip(-1)])
            crops.append(b)
            owner += [k] * len(b)
    return torch.cat(crops), torch.tensor(owner, dtype=torch.long)


@torch.no_grad()
def nearest(q, refs, owner, n_img, exclude_self=None, chunk=4096):
    """
    q    : (n, D) запросы, refs : (M, D) кропы, owner : (M,) номер фото у каждого кропа.
    Возвращает d1, d2 (до двух ближайших РАЗНЫХ фото), номер ближайшего фото и номер лучшего кропа.
    exclude_self : (n,) номер фото, которое нельзя считать соседом (для калибровки на самих фото).
    """
    n = q.shape[0]
    qn = (q * q).sum(1, keepdim=True)
    per_img = torch.full((n, n_img), float("inf"), device=q.device)
    best = torch.full((n,), float("inf"), device=q.device)
    best_idx = torch.zeros(n, dtype=torch.long, device=q.device)
    for s in range(0, refs.shape[0], chunk):
        r = refs[s:s + chunk]
        o = owner[s:s + chunk]
        dd = (qn + (r * r).sum(1)[None, :] - 2.0 * (q @ r.T)).clamp_min(0)   # квадраты расстояний
        if exclude_self is not None:
            dd = dd.masked_fill(exclude_self[:, None] == o[None, :], float("inf"))
        per_img.scatter_reduce_(1, o[None, :].expand(n, -1), dd, reduce="amin")
        m, a = dd.min(1)
        upd = m < best
        best[upd] = m[upd]
        best_idx[upd] = a[upd] + s
    top = per_img.topk(2, dim=1, largest=False)
    d1, d2 = top.values.sqrt().T
    return d1, d2, top.indices[:, 0], best_idx


def main():
    a = parse_args()
    assert len(a.runs) == len(a.data_dirs), "нужно по одной папке с данными на каждый запуск"
    ex_files = a.exclude_files or ["-"] * len(a.runs)
    assert len(ex_files) == len(a.runs), "нужно по одному файлу исключений на каждый запуск"
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

        # ---------- чекпоинты: все ema_*.pt и last.pt, если его шага нет среди снимков
        cks = [(int(p.stem.split("_")[1]), p) for p in sorted(run.glob("ema_*.pt"))]
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        cfg = last["args"]
        model = model_from_checkpoint(last, weights=None).to(dev).eval()
        if last["step"] not in [s for s, _ in cks]:
            cks.append((last["step"], run / "last.pt"))
        cks.sort()
        del last
        S = cfg["img_size"]
        print(f"\n=== {name}: чекпоинты {[s for s, _ in cks]}", flush=True)

        # ---------- обучающие фото и все варианты их кропов
        t0 = time.time()
        files, imgs = load_images(ddir, cfg["pre_size"], exclude)
        boxes = load_boxes(a.boxes, files, imgs) if a.boxes else None
        refs, owner = build_refs(imgs, S, scales, a.npos, boxes, a.nsides)
        refs = refs.flatten(1).to(dev)
        owner = owner.to(dev)
        n_img = len(imgs)
        print(f"фото {n_img}, вариантов кропа {refs.shape[0]} ({refs.shape[0] / n_img:.0f} на фото), "
              f"{time.time() - t0:.1f} с", flush=True)

        # ---------- калибровка на самих фото
        real = center_crops(imgs, S)
        if boxes is not None:   # настоящие кропы с той же случайной схемой, что и при обучении; первые n_img по одному на фото
            torch.manual_seed(2026)
            ds = CatDataset(imgs, S, cfg["min_scale"], repeats=1, boxes=boxes)
            real = torch.stack([ds[k] for _ in range(a.n_real) for k in range(n_img)])
        q = real[:n_img].to(dev).flatten(1)
        b1, b2, _, _ = nearest(q, refs, owner, n_img, exclude_self=torch.arange(n_img, device=dev))
        base_ratio = (b1 / b2).cpu().numpy()
        print(f"настоящие фото: медиана d1 {b1.median():.1f}, ratio<0.5 {np.mean(base_ratio < 0.5) * 100:.1f}%, "
              f"ratio<0.7 {np.mean(base_ratio < 0.7) * 100:.1f}%", flush=True)
        f_real = None if inception is None else inception_feats(inception, real.to(dev))

        # ---------- генерация и поиск соседей для каждого чекпоинта
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
                torch.manual_seed(1000 + i)  # одинаковый шум внутри SDE для всех чекпоинтов
                xs.append(sample(model, sde, noise[i:i + a.batch].to(dev), steps=a.steps,
                                 rho=a.rho if a.method != "sde" else 1.0, method=a.method, amp_ctx=bf16))
            x = torch.cat(xs)
            d1, d2, nn_img, best_idx = nearest(x.flatten(1).float(), refs, owner, n_img)
            r = (d1 / d2).cpu().numpy()

            # 16 самых подозрительных: сэмпл и лучший кроп фото рядом
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
            print(f"шаг {step:7d} | {time.time() - t0:5.1f} с | медиана d1 {row['median_d1']:.1f} | "
                  f"ratio<0.5 {row['ratio_lt_05']:.1f}% | ratio<0.7 {row['ratio_lt_07']:.1f}% | "
                  f"разных ближайших фото {row['unique_nn']} из {n_img}"
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
    print(f"\nготово, результаты в {out}", flush=True)


if __name__ == "__main__":
    main()
