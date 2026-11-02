#!/usr/bin/env python
"""
Диффузионная модель в стиле DDPM (VP-SDE, предсказание шума или скорости v) на фотографиях кошки.

  прямой процесс   x_t = m(t) x_0 + sigma(t) eps,  расписание beta(t) линейное или косинусное (--schedule)
  лосс             || eps_theta(x_t, t) - eps ||^2 или то же для v = m eps - sigma x_0 (--pred),
                   веса Min-SNR (--min-snr),  t ~ U(t_min, 1)
  генерация        обратное ОДУ (Эйлер / Хойн) или обратное SDE (Эйлер-Маруяма)

Архитектура выбирается по имени (--model), ее гиперпараметры заданы в models/__init__.py.

Пример:
  python train.py --data-dir dataset_unique --out-dir runs/cat64 --model unet --resume
Только генерация из чекпоинта:
  python train.py --out-dir runs/cat64 --sample-only --sample-method sde --sample-steps 1000
"""
import argparse
import copy
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2
from torchvision.utils import save_image

from models import CONFIGS, UNet, build_model, config_from_checkpoint  # UNet оставлен для старых импортов

IMG_EXT = {".jpg", ".jpeg", ".png"}

# Дополнительные аугментации, применяются после кропа и отражения к uint8-картинке 3 x S x S.
# Правятся руками. Пример против запоминания:
#   EXTRA_AUG = [v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)]
EXTRA_AUG = []


# ----------------------------------------------------------------------------- аргументы
def parse_args():
    p = argparse.ArgumentParser(description="DDPM / VP-SDE на фото кошки")
    g = p.add_argument_group("данные")
    g.add_argument("--data-dir", type=str, default="dataset_unique")
    g.add_argument("--exclude-file", type=str, default=None, help="файл с именами фото, которые не берем")
    g.add_argument("--img-size", type=int, default=64, help="размер, на котором учимся")
    g.add_argument("--pre-size", type=int, default=128, help="короткая сторона после предварительного уменьшения")
    g.add_argument("--min-scale", type=float, default=0.85, help="минимальная сторона кропа в долях короткой стороны")
    g.add_argument("--boxes", type=str, default=None,
                   help="boxes.json: кроп всегда содержит бокс кошки целиком, --min-scale тогда не используется")
    g.add_argument("--workers", type=int, default=8)

    g = p.add_argument_group("модель")
    g.add_argument("--model", type=str, default="unet", choices=sorted(CONFIGS),
                   help="пресет из models/__init__.py")

    g = p.add_argument_group("диффузия")
    g.add_argument("--beta-min", type=float, default=0.1)
    g.add_argument("--beta-max", type=float, default=20.0)
    g.add_argument("--t-min", type=float, default=1e-3)
    g.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"],
                   help="расписание шума: линейное beta (DDPM) или косинусное (Improved DDPM)")
    g.add_argument("--cond", type=str, default="log_snr", choices=["log_snr", "log_sigma"],
                   help="вход сети: log(sigma/m) или log sigma (старые запуски)")
    g.add_argument("--pred", type=str, default="eps", choices=["eps", "v"],
                   help="что предсказывает сеть: шум eps или скорость v = m eps - sigma x_0")
    g.add_argument("--min-snr", type=float, default=0.0,
                   help="gamma для весов Min-SNR (Hang et al. 2023, обычно 5), 0 = без весов")

    g = p.add_argument_group("обучение")
    g.add_argument("--out-dir", type=str, default="runs/cat64")
    g.add_argument("--steps", type=int, default=100_000)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--lr", type=float, default=2e-4, help="lr в начале косинусного отжига")
    g.add_argument("--lr-min", type=float, default=1e-6, help="lr в конце косинусного отжига")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--ema", type=float, default=0.9995)
    g.add_argument("--amp", type=str, default="auto", choices=["auto", "fp16", "bf16", "none"])
    g.add_argument("--no-compile", action="store_true", help="не использовать torch.compile")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--resume", action="store_true", help="продолжить с out-dir/last.pt, если он есть")

    g = p.add_argument_group("логи и генерация")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--sample-every", type=int, default=5000)
    g.add_argument("--ckpt-every", type=int, default=5000)
    g.add_argument("--snapshot-every", type=int, default=20000, help="отдельные EMA-снимки для сравнения")
    g.add_argument("--n-samples", type=int, default=64)
    g.add_argument("--sample-steps", type=int, default=1000)
    g.add_argument("--sample-method", type=str, default="sde", choices=["euler", "heun", "sde"])
    g.add_argument("--rho", type=float, default=2.0, help="сгущение шагов у t = 0 для ОДУ (для SDE берется 1)")
    g.add_argument("--sample-only", action="store_true")
    g.add_argument("--ckpt", type=str, default=None, help="чекпоинт для --sample-only (по умолчанию out-dir/last.pt)")
    return p.parse_args()


# ----------------------------------------------------------------------------- данные
def load_images(data_dir, pre_size, exclude=()):
    """Читает все фото из корня папки (подпапки не трогает), уменьшает до pre_size по короткой стороне."""
    files = sorted(
        p for p in Path(data_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.name not in exclude
    )
    imgs = []
    for p in files:
        im = Image.open(p)
        im.draft("RGB", (2 * pre_size, 2 * pre_size))  # быстрое декодирование JPEG сразу в уменьшенном виде
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = TF.resize(im, pre_size, interpolation=InterpolationMode.LANCZOS)  # короткая сторона = pre_size
        imgs.append(TF.pil_to_tensor(im))                                       # uint8, (3, H, W)
    return files, imgs


def load_boxes(path, files, imgs):
    """Боксы из boxes.json в пикселях уменьшенных фото, с округлением наружу. Порядок как у files."""
    data = json.loads(Path(path).read_text())
    out = []
    for p, img in zip(files, imgs):
        if p.name not in data:
            raise SystemExit(f"в {path} нет бокса для {p.name}")
        r = data[p.name]
        b = r["box"]
        _, H, W = img.shape
        kx, ky = W / r["width"], H / r["height"]
        out.append((max(math.floor(b["x1"] * kx), 0), max(math.floor(b["y1"] * ky), 0),
                    min(math.ceil(b["x2"] * kx), W), min(math.ceil(b["y2"] * ky), H)))
    return out


def min_crop_side(box, H, W, size):
    """Наименьшая сторона кропа: вмещает бокс и не заставляет растягивать кроп до size. Бокс должен влезать в фото."""
    return min(max(box[2] - box[0], box[3] - box[1], size), H, W)


def crop_offsets(box, H, W, side):
    """top_min, top_max, left_min, left_max: где может стоять квадрат со стороной side, чтобы бокс был внутри."""
    x1, y1, x2, y2 = box
    top_max, left_max = min(y1, H - side), min(x1, W - side)
    return min(max(0, y2 - side), top_max), top_max, min(max(0, x2 - side), left_max), left_max


# uint8 [0, 255] -> float [-1, 1]
TO_UNIT = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])


def to_unit(x_uint8):
    return TO_UNIT(x_uint8).as_subclass(torch.Tensor)


class CatDataset(Dataset):
    """
    Случайный квадратный кроп + уменьшение до img_size + случайное отражение (+ EXTRA_AUG).
    С boxes кроп всегда содержит бокс кошки: сторона случайна от наименьшей допустимой до короткой стороны фото,
    сдвиг случаен в пределах, где бокс остается внутри.
    """

    def __init__(self, imgs, size, min_scale, repeats=50, boxes=None):
        self.imgs, self.size, self.min_scale, self.repeats, self.boxes = imgs, size, min_scale, repeats, boxes
        self.aug = v2.Compose([v2.RandomHorizontalFlip(p=0.5), *EXTRA_AUG])

    def __len__(self):
        return len(self.imgs) * self.repeats

    def __getitem__(self, i):
        img = self.imgs[i % len(self.imgs)]
        _, H, W = img.shape
        short = min(H, W)
        if self.boxes is None:
            # квадрат со стороной от min_scale до 1 от короткой стороны, в случайном месте
            side = int(short * (self.min_scale + (1 - self.min_scale) * torch.rand(1).item()))
            top = torch.randint(0, H - side + 1, (1,)).item()
            left = torch.randint(0, W - side + 1, (1,)).item()
        else:
            box = self.boxes[i % len(self.imgs)]
            side = torch.randint(min_crop_side(box, H, W, self.size), short + 1, (1,)).item()
            top_min, top_max, left_min, left_max = crop_offsets(box, H, W, side)
            top = torch.randint(top_min, top_max + 1, (1,)).item()
            left = torch.randint(left_min, left_max + 1, (1,)).item()
        crop = TF.resized_crop(img, top, left, side, side, [self.size, self.size], antialias=True)
        return to_unit(self.aug(crop))


def center_crops(imgs, size):
    """Центральные квадратные кропы: эталон для проверки на запоминание. (N, 3, size, size) в [-1, 1]."""
    out = [TF.resize(TF.center_crop(img, min(img.shape[-2:])), [size, size], antialias=True) for img in imgs]
    return to_unit(torch.stack(out))


def save_grid(x, path, nrow=8, pad=2, upscale=2):
    """x: (N, 3, H, W) в [-1, 1] -> png-сетка, каждая картинка увеличена в upscale раз без сглаживания."""
    x = (x.detach().float().clamp(-1, 1) + 1) / 2
    if upscale > 1:
        x = F.interpolate(x, scale_factor=upscale, mode="nearest")
    save_image(x.cpu(), path, nrow=nrow, padding=pad, pad_value=1.0)


# ----------------------------------------------------------------------------- диффузия
class VPSDE:
    """
    Прямой процесс x_t = m(t) x_0 + sigma(t) eps,  m = exp(-1/2 int_0^t beta),  sigma^2 = 1 - m^2.

    schedule = как растет шум:
      linear : beta(t) = beta_min + t (beta_max - beta_min), как в DDPM.
      cosine : m(t) = cos(theta(t)) / cos(theta_0), как в Improved DDPM (Nichol, Dhariwal 2021).
               theta идет линейно от theta_0 = pi/2 * s/(1+s) до угла, где m = m_end.
               m_end берем как у linear в t = 1, чтобы шум в конце был тот же.
               Шум нарастает плавнее, и средним уровням шума достается больше обучения.
    cond = что сеть получает на вход как «уровень шума»:
      log_sigma : c = log sigma(t). Старый вариант, при t > 0.5 почти не меняется.
      log_snr   : c = log(sigma / m) = -1/2 log SNR. Растет на всем отрезке.
    pred = что предсказывает сеть:
      eps : шум eps, как в DDPM.
      v   : скорость v = m eps - sigma x_0 (Salimans, Ho 2022). При малом шуме это почти eps, при сильном почти -x_0.
    """

    def __init__(self, beta_min=0.1, beta_max=20.0, t_min=1e-3, cond="log_snr", schedule="linear", s=0.008, pred="eps"):
        self.beta_min, self.beta_max, self.t_min, self.cond, self.schedule = beta_min, beta_max, t_min, cond, schedule
        self.pred = pred
        if schedule == "cosine":
            m_end = math.exp(-0.5 * (beta_min + 0.5 * (beta_max - beta_min)))
            self.th0 = 0.5 * math.pi * s / (1 + s)
            self.th1 = math.acos(m_end * math.cos(self.th0))
        elif schedule != "linear":
            raise ValueError(schedule)

    @staticmethod
    def _t(t):
        # считаем в float64: при малых t int_beta порядка 1e-5, и float32 теряет точность
        return t.double() if torch.is_tensor(t) else torch.tensor(float(t), dtype=torch.float64)

    @staticmethod
    def _out(v, t):
        return v.to(t.dtype) if torch.is_tensor(t) else v.float()

    def beta(self, t):
        if self.schedule == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)
        th = self.th0 + self._t(t) * (self.th1 - self.th0)
        return self._out(2 * (self.th1 - self.th0) * torch.tan(th), t)  # d/dt int_beta

    def int_beta(self, t):
        if self.schedule == "linear":
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        th = self.th0 + self._t(t) * (self.th1 - self.th0)
        return self._out(-2 * torch.log(torch.cos(th) / math.cos(self.th0)), t)

    def m(self, t):
        return torch.exp(-0.5 * self.int_beta(t))

    def sigma(self, t):
        return torch.sqrt(-torch.expm1(-self.int_beta(t)))  # sqrt(1 - m^2), точно и при малых t

    def c(self, t):
        """Вход сети, кодирующий момент t."""
        if self.cond == "log_sigma":
            return torch.log(self.sigma(t))
        return 0.5 * torch.log(torch.expm1(self.int_beta(t)))  # log(sigma / m), точно и при малых t

    def to_eps(self, out, xt, t):
        """Выход сети -> предсказанный шум. t : (B,). Для v: eps = m v + sigma x_t, так как m^2 + sigma^2 = 1."""
        if self.pred == "eps":
            return out
        return self.m(t).view(-1, 1, 1, 1) * out + self.sigma(t).view(-1, 1, 1, 1) * xt


def sde_from_cfg(cfg):
    """VPSDE по аргументам из чекпоинта. Старые чекпоинты: log_sigma, linear и eps."""
    return VPSDE(cfg["beta_min"], cfg["beta_max"], cfg["t_min"],
                 cfg.get("cond", "log_sigma"), cfg.get("schedule", "linear"), pred=cfg.get("pred", "eps"))


def diffusion_loss(net, sde, x0, min_snr=0.0):
    """
    Algorithm 1 из DDPM: шумим по формуле и учим угадывать шум или скорость v.
    Min-SNR: вес ошибки x_0 равен min(SNR, gamma), в пересчете на лосс по eps или v это веса ниже.
    Возвращает лосс для обучения, t и eps-MSE каждой картинки. eps-MSE идет только в логи: он без весов
    и в одном масштабе при любом --pred (ошибка v = ошибка eps / m), поэтому запуски можно сравнивать.
    """
    B = x0.shape[0]
    t = sde.t_min + (1 - sde.t_min) * torch.rand(B, device=x0.device)
    eps = torch.randn_like(x0)
    m = sde.m(t).view(B, 1, 1, 1)
    s = sde.sigma(t).view(B, 1, 1, 1)
    xt = m * x0 + s * eps
    pred = net(xt, sde.c(t))
    target = m * eps - s * x0 if sde.pred == "v" else eps
    per_sample = ((pred.float() - target) ** 2).mean(dim=(1, 2, 3))
    eps_mse = per_sample * m.view(B) ** 2 if sde.pred == "v" else per_sample
    if min_snr > 0:
        snr = (m / s).view(B) ** 2
        per_sample = per_sample * snr.clamp(max=min_snr) / (snr + 1 if sde.pred == "v" else snr)
    return per_sample.mean(), t, eps_mse


@torch.no_grad()
def sample(net, sde, x_start, steps=50, rho=2.0, method="heun", amp_ctx=nullcontext):
    """Обратный процесс от t = 1 до t_min, затем финальное расшумление x0 = (x - sigma eps) / m."""
    x = x_start.clone().float()
    n, dev = x.shape[0], x.device
    u = np.linspace(1.0, 0.0, steps + 1)
    ts = sde.t_min + (1.0 - sde.t_min) * u ** rho

    def eps_at(x, t):
        tt = torch.full((n,), float(t), device=dev)
        with amp_ctx():
            e = net(x, sde.c(tt))
        return sde.to_eps(e.float(), x, tt)

    def drift(x, t, k=1.0):
        # правая часть при движении назад: x <- x + h * drift;  k = 1 для ОДУ, k = 2 для SDE
        tt = torch.tensor(float(t), device=dev)
        s = -eps_at(x, t) / sde.sigma(tt)
        return 0.5 * sde.beta(tt) * (x + k * s)

    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        h = t - tn
        if method == "euler":
            x = x + h * drift(x, t)
        elif method == "heun":
            d = drift(x, t)
            x_e = x + h * d
            x = x + 0.5 * h * (d + drift(x_e, tn))
        elif method == "sde":
            x = x + h * drift(x, t, k=2.0)
            if i < steps - 1:
                x = x + math.sqrt(float(sde.beta(float(t))) * h) * torch.randn_like(x)
        else:
            raise ValueError(method)

    tt = torch.full((n,), float(ts[-1]), device=dev)
    x0 = (x - sde.sigma(tt).view(-1, 1, 1, 1) * eps_at(x, ts[-1])) / sde.m(tt).view(-1, 1, 1, 1)
    return x0.clamp(-1, 1)


# ----------------------------------------------------------------------------- утилиты
@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p.detach(), 1.0 - decay)


@torch.no_grad()
def nn_check(samples, ref_all, n_ref):
    """Для каждого сэмпла ближайшая обучающая картинка (с учетом отражений) и расстояние в пикселях."""
    d = torch.cdist(samples.flatten(1).float(), ref_all.flatten(1).float())
    dmin, idx = d.min(dim=1)
    return dmin, idx % n_ref


@torch.no_grad()
def train_nn_baseline(ref, ref_all):
    """Типичное расстояние от обучающей картинки до ближайшей другой (не себя и не своего отражения)."""
    n = ref.shape[0]
    d = torch.cdist(ref.flatten(1).float(), ref_all.flatten(1).float())
    ar = torch.arange(n, device=ref.device)
    d[ar, ar] = float("inf")
    d[ar, ar + n] = float("inf")
    return d.min(dim=1).values


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_amp(amp, device):
    if device != "cuda" or amp == "none":
        return "none", nullcontext
    if amp == "auto":
        amp = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"  # V100 = 7.0 -> fp16
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return amp, (lambda: torch.autocast("cuda", dtype=dtype))


def cosine_lr(step, total, lr, lr_min):
    """Косинусный отжиг без прогрева: lr на шаге 0, lr_min на шаге total."""
    p = min(step, total) / max(total, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * p))


def run_sampling(ema, sde, x_start, args, amp_ctx, ref, ref_all, path_prefix, method=None, steps=None):
    method = method or args.sample_method
    steps = steps or args.sample_steps
    rho = 1.0 if method == "sde" else args.rho
    t0 = time.time()
    x = sample(ema, sde, x_start, steps=steps, rho=rho, method=method, amp_ctx=amp_ctx)
    dt = time.time() - t0
    save_grid(x, f"{path_prefix}.png")

    dmin, idx = nn_check(x, ref_all, ref.shape[0])
    k = min(32, x.shape[0])
    pairs = torch.stack([x[:k], ref[idx[:k]]], dim=1).flatten(0, 1)  # сэмпл, ближайшая обучающая, сэмпл, ...
    save_grid(pairs, f"{path_prefix}_nn.png", nrow=8)
    return x, dmin, dt


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    out = Path(args.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    amp, amp_ctx = make_amp(args.amp, device)

    print(f"torch {torch.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "") + f" | amp {amp}", flush=True)

    # ---- данные
    exclude = set()
    if args.exclude_file:
        for line in Path(args.exclude_file).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                exclude.add(line)
    t0 = time.time()
    files, imgs = load_images(args.data_dir, args.pre_size, exclude)
    n_port = sum(im.shape[1] > im.shape[2] for im in imgs)
    print(f"фото: {len(imgs)}, вертикальных {n_port}, горизонтальных {len(imgs) - n_port}, "
          f"загрузка {time.time() - t0:.1f} с", flush=True)
    if len(imgs) == 0:
        raise SystemExit(f"в {args.data_dir} не нашлось фото")

    boxes = load_boxes(args.boxes, files, imgs) if args.boxes else None
    ds = CatDataset(imgs, args.img_size, args.min_scale, boxes=boxes)
    save_grid(torch.stack([ds[i * 7 % len(ds)] for i in range(64)]), out / "data_preview.png")

    ref = center_crops(imgs, args.img_size).to(device)
    ref_all = torch.cat([ref, ref.flip(-1)])
    base_nn = train_nn_baseline(ref, ref_all)
    print(f"расстояние между ближайшими обучающими картинками: медиана {base_nn.median():.2f}, "
          f"10% квантиль {base_nn.quantile(0.1):.2f}", flush=True)

    # ---- модель
    sde = sde_from_cfg(vars(args))
    ckpt_path = Path(args.ckpt) if args.ckpt else out / "last.pt"
    ck = None
    if (args.resume or args.sample_only) and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    elif args.sample_only:
        raise SystemExit(f"нет чекпоинта {ckpt_path}")

    model, model_cfg = build_model(args.model, args.img_size)
    if ck is not None:
        for key, old in (("cond", "log_sigma"), ("schedule", "linear"), ("pred", "eps")):
            ck_val = ck["args"].get(key, old)
            if ck_val != getattr(args, key):
                raise SystemExit(f"чекпоинт обучен с --{key} {ck_val}, а сейчас --{key} {getattr(args, key)}")
        old_cfg = config_from_checkpoint(ck)
        if old_cfg != model_cfg:
            raise SystemExit(f"конфигурация модели в чекпоинте отличается от --model {args.model}:\n"
                             f"  в чекпоинте {old_cfg}\n  сейчас      {model_cfg}")
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"модель {args.model}: {n_par / 1e6:.1f} M параметров | {model_cfg}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, args.steps, args.lr, args.lr_min) / args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))

    start_step = 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if not args.sample_only:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"]
        print(f"загружен чекпоинт {ckpt_path}, шаг {start_step}", flush=True)
        del ck

    # один и тот же стартовый шум на всех проверках
    fixed_noise = torch.randn(args.n_samples, 3, args.img_size, args.img_size,
                              generator=torch.Generator().manual_seed(1234)).to(device)

    if args.sample_only:
        prefix = out / f"final_{args.sample_method}_{args.sample_steps}_step{start_step}"
        _, dmin, dt = run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all, prefix)
        print(f"готово за {dt:.1f} с: {prefix}.png | NN-расстояние медиана {dmin.median():.2f} "
              f"(у обучающих {base_nn.median():.2f})", flush=True)
        return

    (out / "config.json").write_text(json.dumps({**vars(args), "model_cfg": model_cfg}, indent=2, ensure_ascii=False))

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=(device == "cuda"), persistent_workers=args.workers > 0)

    def infinite():
        while True:
            for b in loader:
                yield b

    it = infinite()
    use_compile = device == "cuda" and not args.no_compile
    net = torch.compile(model) if use_compile else model
    if use_compile:
        print("torch.compile включен, первые шаги будут медленными, пока идет компиляция", flush=True)

    log_f = open(out / "log.csv", "a", newline="")
    log_w = csv.writer(log_f)
    nn_f = open(out / "nn_stats.csv", "a", newline="")
    nn_w = csv.writer(nn_f)
    if start_step == 0:
        log_w.writerow(["step", "loss", "loss_t0-0.2", "loss_t0.2-0.4", "loss_t0.4-0.6",
                        "loss_t0.6-0.8", "loss_t0.8-1", "lr", "grad_norm", "it_per_s"])
        nn_w.writerow(["step", "nn_median", "nn_min", "train_nn_median", "sample_seconds"])

    n_bins = 5
    edges = torch.linspace(0, 1, n_bins + 1, device=device)[1:-1]
    loss_sum = torch.zeros((), device=device)
    bin_sum = torch.zeros(n_bins, device=device)
    bin_cnt = torch.zeros(n_bins, device=device)
    gnorm_sum = 0.0
    n_acc = 0
    t_log = time.time()
    t_start = time.time()

    def checkpoint(s):
        return {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": s,
                "args": vars(args), "model_name": args.model, "model_cfg": model_cfg}

    model.train()
    for step in range(start_step, args.steps):
        x0 = next(it).to(device, non_blocking=True)

        with amp_ctx():
            loss, t, per = diffusion_loss(net, sde, x0, args.min_snr)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))

        # статистика: в логи пишем eps-MSE без весов, он сравним между запусками
        loss_sum += per.detach().mean()
        b = torch.bucketize(t, edges)
        bin_sum.index_add_(0, b, per.detach())
        bin_cnt.index_add_(0, b, torch.ones_like(per))
        g = float(gnorm)
        if math.isfinite(g):  # в fp16 GradScaler иногда пропускает шаг с inf, его не считаем
            gnorm_sum += g
        n_acc += 1

        s = step + 1
        if s % args.log_every == 0 or s == args.steps:
            ips = n_acc / (time.time() - t_log)
            bins = (bin_sum / bin_cnt.clamp(min=1)).tolist()
            lr = sched.get_last_lr()[0]
            l = (loss_sum / n_acc).item()
            log_w.writerow([s, f"{l:.5f}"] + [f"{v:.5f}" for v in bins]
                           + [f"{lr:.2e}", f"{gnorm_sum / n_acc:.3f}", f"{ips:.2f}"])
            log_f.flush()
            eta = (args.steps - s) / max(ips, 1e-9) / 3600
            print(f"step {s:7d} | loss {l:.4f} | по t: " + " ".join(f"{v:.3f}" for v in bins)
                  + f" | lr {lr:.1e} | grad {gnorm_sum / n_acc:.2f} | {ips:.1f} it/s | осталось ~{eta:.1f} ч",
                  flush=True)
            loss_sum.zero_(); bin_sum.zero_(); bin_cnt.zero_()
            gnorm_sum, n_acc, t_log = 0.0, 0, time.time()

        if s % args.sample_every == 0 or s == args.steps:
            _, dmin, dt = run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all,
                                       out / "samples" / f"step_{s:07d}")
            nn_w.writerow([s, f"{dmin.median():.3f}", f"{dmin.min():.3f}", f"{base_nn.median():.3f}", f"{dt:.1f}"])
            nn_f.flush()
            print(f"  сэмплы сохранены ({dt:.1f} с) | NN-расстояние медиана {dmin.median():.2f}, "
                  f"мин {dmin.min():.2f} (у обучающих {base_nn.median():.2f})", flush=True)
            t_log = time.time()  # время генерации не входит в it/s

        if s % args.ckpt_every == 0 or s == args.steps:
            atomic_save(checkpoint(s), out / "last.pt")
        if args.snapshot_every and s % args.snapshot_every == 0:
            torch.save({"ema": ema.state_dict(), "step": s, "args": vars(args),
                        "model_name": args.model, "model_cfg": model_cfg}, out / f"ema_{s:07d}.pt")

    # финальные картинки: детерминированное ОДУ и стохастическое SDE
    if start_step < args.steps:
        run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all, out / "final_heun50", "heun", 50)
        run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all, out / "final_sde1000", "sde", 1000)
    print(f"обучение завершено за {(time.time() - t_start) / 3600:.2f} ч", flush=True)
    log_f.close()
    nn_f.close()


if __name__ == "__main__":
    main()
