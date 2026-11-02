#!/usr/bin/env python
"""
Диагностика обученной модели без вмешательства в обучение.

1) denoise.png  : настоящие фото -> шум до момента t -> одно предсказание сети -> денойзер
                  D = (x_t - sigma * eps_theta) / m.  Строки: чистые фото, затем для каждого t пара строк
                  (зашумленная картинка, восстановленная).
2) samplers.png : один и тот же стартовый шум, разные способы решать обратный процесс.
                  Каждая строка = один сэмплер (порядок печатается в консоль).

Пример:
  srun --jobid=139160 --overlap -n 1 -c 2 .venv/bin/python diagnose.py \
      --run runs/cat64_fast --data-dir Terra --exclude-file exclude.txt
"""
import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from models import model_from_checkpoint
from train import sde_from_cfg, center_crops, load_images, sample, save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None, help="по умолчанию run/last.pt")
    p.add_argument("--data-dir", type=str, default="Terra")
    p.add_argument("--exclude-file", type=str, default=None)
    p.add_argument("--n", type=int, default=8, help="картинок в строке")
    a = p.parse_args()

    run = Path(a.run)
    ck = torch.load(a.ckpt or run / "last.pt", map_location="cpu", weights_only=False)
    cfg = ck["args"]
    device = "cuda"

    model = model_from_checkpoint(ck, weights="ema")
    model = model.to(device).eval()
    sde = sde_from_cfg(cfg)
    S = cfg["img_size"]
    step = ck["step"]
    print(f"чекпоинт: шаг {step}")

    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    fp32 = nullcontext

    # ------------------------------------------------------------------ 1. расшумление за один шаг
    exclude = set()
    if a.exclude_file:
        for line in Path(a.exclude_file).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                exclude.add(line)
    _, imgs = load_images(a.data_dir, cfg["pre_size"], exclude)
    idx = torch.linspace(0, len(imgs) - 1, a.n).long().tolist()   # равномерно по датасету
    x0 = center_crops([imgs[i] for i in idx], S).to(device)

    g = torch.Generator(device=device).manual_seed(0)
    eps = torch.randn(x0.shape, generator=g, device=device)
    rows = [x0]
    print(f"\nрасшумление за один шаг (fp32, вход сети: {sde.cond}):")
    print(f"  x0-MSE = (sigma/m)^2 * eps-MSE, ошибка восстановленной картинки.")
    print(f"  Для сравнения: если всегда отвечать средним фото, x0-MSE = {x0.var(0).mean().item():.3f}")
    with torch.no_grad():
        for t_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            t = torch.full((a.n,), t_val, device=device)
            m = sde.m(t).view(-1, 1, 1, 1)
            s = sde.sigma(t).view(-1, 1, 1, 1)
            xt = m * x0 + s * eps
            e = sde.to_eps(model(xt, sde.c(t)).float(), xt, t)
            D_raw = (xt - s * e) / m
            D = D_raw.clamp(-1, 1)
            mse = ((e - eps) ** 2).mean().item()
            x0_mse = ((D_raw - x0) ** 2).mean().item()
            dc = ((D_raw.mean((2, 3)) - x0.mean((2, 3))) ** 2).mean().sqrt().item()  # ошибка среднего цвета
            print(f"  t={t_val:<4}  sigma={s[0].item():.3f}  m={m[0].item():.3f}  c={sde.c(t)[0].item():6.3f}  "
                  f"eps-MSE={mse:.4f}  x0-MSE={x0_mse:.3f}  ошибка среднего цвета={dc:.3f}")
            rows += [xt.clamp(-1, 1), D]
    save_grid(torch.cat(rows), run / f"diag_denoise_step{step}.png", nrow=a.n)

    # ------------------------------------------------------------------ 2. сравнение сэмплеров
    noise = torch.randn(a.n, 3, S, S, generator=torch.Generator().manual_seed(1234)).to(device)
    configs = [
        ("heun 50 шагов, bf16 (как в обучении)", "heun", 50, 2.0, bf16),
        ("heun 50 шагов, fp32", "heun", 50, 2.0, fp32),
        ("heun 200 шагов, fp32", "heun", 200, 2.0, fp32),
        ("euler 500 шагов, fp32", "euler", 500, 1.0, fp32),
        ("SDE 1000 шагов, fp32", "sde", 1000, 1.0, fp32),
    ]
    rows = []
    print("\nсэмплеры (строки samplers.png сверху вниз):")
    for name, method, steps, rho, ctx in configs:
        t0 = time.time()
        torch.manual_seed(0)  # одинаковый шум внутри SDE между запусками
        x = sample(model, sde, noise, steps=steps, rho=rho, method=method, amp_ctx=ctx)
        rows.append(x)
        print(f"  {name:40s} {time.time() - t0:5.1f} с")
    save_grid(torch.cat(rows), run / f"diag_samplers_step{step}.png", nrow=a.n)
    print(f"\nкартинки сохранены в {run}")


if __name__ == "__main__":
    main()
