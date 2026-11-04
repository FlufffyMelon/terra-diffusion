#!/usr/bin/env python
"""
Качество латентной диффузии: FD-DINOv2 и KID между сэмплами и настоящими кадрами.

  FD-DINOv2 : расстояние Фреше между гауссианами признаков DINOv2 ViT-L/14 (CLS после LayerNorm, вход 224),
              как в Stein et al. 2023. Чувствителен к общему сдвигу и разбросу, но смещен при малом N:
              сравнивать можно только при одинаковом числе сэмплов.
  KID       : несмещенная оценка MMD^2 с полиномиальным ядром на признаках Inception-v3 (как в memorization.py),
              работает и на тысяче сэмплов, дает разброс по подвыборкам.
  Бесплатно : FID на Inception и KID на DINOv2, признаки все равно посчитаны.

Эталон: --n-real кадров fit_1x1 из манифеста, по одному кропу с тем же правилом, что при обучении (бокс внутри).
Две опорные строки, чтобы понимать шкалу:
  настоящие      : другие --n-gen настоящих кадров. Нижняя граница метрик при таком N.
  реконструкция  : те же кадры через VAE. Лучшее, что можно получить с этим VAE.

Строки: чекпоинт или чекпоинт|слабая модель|w (autoguidance), все через SDE --steps шагов.
  .venv/bin/python tools/eval_metrics.py --dataset ~/DiffusionLearn_data/pretrain_v1 --out runs/eval/v2_v3 \\
      --rows runs/ldm_v3/ckpt/last.ckpt "runs/ldm_v3/ckpt/last.ckpt|runs/ldm_v2/ckpt/last.ckpt|2"
В --out: metrics.csv (строка на вариант, дописывается по мере готовности), feats/*.npz, grids/*.jpg.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from scipy import linalg
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import FrameCrops, read_manifest
from memorization import inception_feats, kid, load_inception
from models import UNet
from train import VPSDE, sample

DINO = "facebook/dinov2-large"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--n-real", type=int, default=10000)
    p.add_argument("--n-gen", type=int, default=5000)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-baselines", action="store_true", help="без строк «настоящие» и «реконструкция»")
    return p.parse_args()


class Guided(torch.nn.Module):
    """Autoguidance: out = bad + w (good - bad)."""

    def __init__(self, good, bad, w):
        super().__init__()
        self.good, self.bad, self.w = good, bad, w

    def forward(self, x, c):
        g, b = self.good(x, c), self.bad(x, c)
        return b + self.w * (g - b)


def load_unet(path, dev):
    """EMA-веса из ckpt/last.ckpt или ema_*.pt; конфигурация из config.json того же запуска."""
    path = Path(path)
    run = path.parent.parent if path.parent.name == "ckpt" else path.parent
    cfg = json.loads((run / "config.json").read_text())
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["ema"] if "ema" in ck else {k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    m = UNet(img_size=64, in_ch=4, base=cfg["base"], mults=[int(v) for v in cfg["mults"].split(",")], num_res=cfg["num_res"],
             attn_res=[int(v) for v in cfg["attn_res"].split(",")], dropout=cfg["dropout"], norm_groups=32,
             head_ch=cfg.get("head_ch") or None, qk_norm=cfg.get("qk_norm", False))
    m.load_state_dict(sd)
    step = ck.get("step", ck.get("global_step"))
    return m.to(dev).eval().requires_grad_(False), cfg, f"{run.name}-{step}"


class Features:
    """DINOv2 и Inception для картинок (B, 3, S, S) в [-1, 1] на GPU."""

    def __init__(self, dev):
        self.dino = AutoModel.from_pretrained(DINO, local_files_only=True).to(dev).eval()
        self.inc = load_inception(dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        b = (x.float().clamp(-1, 1) + 1) / 2
        b = F.interpolate(b, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        d = self.dino(pixel_values=(b - self.mean) / self.std).pooler_output.double().cpu()
        return d, inception_feats(self.inc, x)


def fd(a, b):
    """Расстояние Фреше между гауссианами двух наборов признаков (N, d)."""
    a, b = a.numpy(), b.numpy()
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr sqrt(S1 S2) = tr sqrt(sqrt(S1) S2 sqrt(S1)): вторая матрица симметричная, ее корень через eigvalsh устойчив
    e, v = linalg.eigh(s1)
    r1 = (v * np.sqrt(np.clip(e, 0, None))) @ v.T
    tr_covmean = np.sqrt(np.clip(linalg.eigvalsh(r1 @ s2 @ r1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def real_loader(rows, root, idx, a):
    ds = Subset(FrameCrops(rows, root, size=512, crops=1, seed=a.seed), idx)
    return DataLoader(ds, batch_size=a.batch, num_workers=a.workers, prefetch_factor=4)


def main():
    a = parse_args()
    dev = "cuda"
    out = Path(a.out)
    (out / "feats").mkdir(parents=True, exist_ok=True)
    (out / "grids").mkdir(exist_ok=True)
    root = Path(a.dataset)
    rows = read_manifest(root / "manifest.jsonl")
    perm = np.random.default_rng(a.seed).permutation(len(rows))
    real_idx, base_idx = perm[:a.n_real].tolist(), perm[a.n_real:a.n_real + a.n_gen].tolist()
    feat = Features(dev)
    t_start = time.time()

    # ------------------------------------------------------------- эталон
    ref_path = out / "feats" / "real.npz"
    if ref_path.exists():
        z = np.load(ref_path)
        ref_d, ref_i = torch.from_numpy(z["dino"]), torch.from_numpy(z["inc"])
    else:
        fd_, fi_ = [], []
        for x, _, ok in real_loader(rows, root, real_idx, a):
            d, i = feat(x[ok, 0].to(dev))
            fd_.append(d); fi_.append(i)
        ref_d, ref_i = torch.cat(fd_), torch.cat(fi_)
        np.savez(ref_path, dino=ref_d.numpy(), inc=ref_i.numpy())
    print(f"эталон: {len(ref_d)} кадров, {time.time() - t_start:.0f} с", flush=True)

    csv_path = out / "metrics.csv"
    new = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["вариант", "n", "fd_dinov2", "kid_inception_x1000", "kid_inception_std_x1000", "fid_inception",
                    "kid_dinov2_x1000", "секунд", "картинок_в_секунду"])

    def report(name, d, i, sec):
        k, ks = kid(ref_i, i)
        kd, _ = kid(ref_d, d)
        r = [name, len(d), f"{fd(ref_d, d):.2f}", f"{k * 1e3:.3f}", f"{ks * 1e3:.3f}", f"{fd(ref_i, i):.2f}", f"{kd * 1e3:.3f}",
             f"{sec:.0f}", f"{len(d) / max(sec, 1e-9):.2f}"]
        w.writerow(r)
        f.flush()
        print(f"{name}: FD-DINOv2 {r[2]} | KID x1000 {r[3]} ± {r[4]} | FID {r[5]} | KID-DINOv2 x1000 {r[6]} | {r[7]} с", flush=True)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True, low_cpu_mem_usage=False).to(dev).eval()

    # ------------------------------------------------------------- опорные строки
    if not a.no_baselines:
        t0 = time.time()
        acc = {"настоящие": ([], []), "реконструкция VAE": ([], [])}
        for x, _, ok in real_loader(rows, root, base_idx, a):
            x = x[ok, 0].to(dev)
            with torch.no_grad():
                # частями по 16: энкодер VAE на 64 картинках 512x512 не помещается в 24 ГБ
                rec = torch.cat([vae.decode(vae.encode(x[j:j + 16]).latent_dist.mean).sample for j in range(0, len(x), 16)])
            for key, img in (("настоящие", x), ("реконструкция VAE", rec)):
                d, i = feat(img)
                acc[key][0].append(d); acc[key][1].append(i)
        sec = time.time() - t0
        for key, (d, i) in acc.items():
            report(key, torch.cat(d), torch.cat(i), sec)

    # ------------------------------------------------------------- варианты
    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    for spec in a.rows:
        parts = spec.split("|")
        good, cfg, gname = load_unet(parts[0], dev)
        net, name = good, f"{gname}, sde {a.steps}"
        if len(parts) == 3:
            bad, _, bname = load_unet(parts[1], dev)
            net, name = Guided(good, bad, float(parts[2])), f"{gname}, ag {bname}, w {parts[2]}, sde {a.steps}"
        sde = VPSDE(pred=cfg["pred"])
        t0 = time.time()
        fd_, fi_ = [], []
        for b in range(0, a.n_gen, a.batch):
            n = min(a.batch, a.n_gen - b)
            s = a.seed * 1_000_003 + b                                  # у всех вариантов одни и те же шумы
            x0 = torch.randn(n, 4, 64, 64, generator=torch.Generator().manual_seed(s)).to(dev)
            torch.manual_seed(s)
            with torch.no_grad():
                zz = sample(net, sde, x0, steps=a.steps, rho=1.0, method="sde", amp_ctx=bf16, clip=False)
                img = torch.cat([vae.decode(zz[j:j + 16].float() / cfg["scale"]).sample for j in range(0, n, 16)])
            if b == 0:
                save_image((img[:32].clamp(-1, 1) + 1) / 2, out / "grids" / f"{name.replace(', ', '_').replace(' ', '')}.jpg", nrow=8)
            d, i = feat(img)
            fd_.append(d); fi_.append(i)
            if b // a.batch % 10 == 0:
                done = b + n
                print(f"  {name}: {done} из {a.n_gen}, {done / (time.time() - t0):.2f} картинок/с", flush=True)
        d, i = torch.cat(fd_), torch.cat(fi_)
        np.savez(out / "feats" / f"{name.replace(', ', '_').replace(' ', '')}.npz", dino=d.numpy(), inc=i.numpy())
        report(name, d, i, time.time() - t0)
        del net, good
        torch.cuda.empty_cache()
    print(f"готово за {(time.time() - t_start) / 3600:.1f} ч", flush=True)


if __name__ == "__main__":
    main()
