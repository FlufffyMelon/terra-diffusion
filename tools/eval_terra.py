#!/usr/bin/env python
"""
Насколько модель с условиями рисует Терру: похожесть, новизна, разнообразие. Все на признаках DINOv2 (как в prepare_cond.py).

  похожесть  : логистическая регрессия «Терра / другая кошка» на обучающих кадрах Терры и случайных других кошках.
               Ее точность на отложенных кадрах terra_holdout.json печатается как проверка. Метрика: доля сэмплов,
               которые она считает Террой (p > 0.5).
  новизна    : косинусная близость каждого сэмпла к ближайшему обучающему кадру Терры. Порог «почти копия» это
               95-й перцентиль той же величины у отложенных кадров: настоящее новое фото Терры похоже на обучающие
               не сильнее. Метрика: доля сэмплов выше порога (у настоящих новых фото около 5%).
  разнообразие и реализм : KID и FD между сэмплами и всеми кадрами Терры, признаки DINOv2. FD при малом N смещен,
               сравнивать только варианты с одинаковым N.

Варианты: строки вида "w=3" или "w=3,type=terra" или "w=3,ag=runs/ldm_v4/ema_0005000.pt@2".
  w     : сила CFG по условиям от ответа без условий
  type  : none (по умолчанию, тип «пусто»), terra (основной кластер Терры) или номер кластера
  ag    : autoguidance слабой моделью с теми же условиями, путь@w
.venv/bin/python tools/eval_terra.py --ckpt runs/ldm_v4/ema_0020000.pt --dataset ~/DiffusionLearn_data/pretrain_v1 \\
    --out runs/eval/terra_v4 --rows w=1 w=2 w=3 "w=3,type=terra"
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
from torchvision.utils import save_image
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import FrameCrops, read_manifest
from guidance import CFG, Guided
from memorization import kid
from models import UNet
from tools.eval_metrics import fd
from train import VPSDE, sample

DINO = "facebook/dinov2-large"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="ckpt/last.ckpt или ema_*.pt модели с условиями")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--n", type=int, default=512, help="сэмплов на вариант")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_unet(path, dev):
    path = Path(path)
    run = path.parent.parent if path.parent.name == "ckpt" else path.parent
    cfg = json.loads((run / "config.json").read_text())
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["ema"] if "ema" in ck else {k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    m = UNet(img_size=64, in_ch=4, base=cfg["base"], mults=[int(v) for v in cfg["mults"].split(",")], num_res=cfg["num_res"],
             attn_res=[int(v) for v in cfg["attn_res"].split(",")], dropout=cfg["dropout"], norm_groups=32,
             head_ch=cfg.get("head_ch") or None, qk_norm=cfg.get("qk_norm", False), cond_sizes=cfg.get("cond_sizes", []))
    m.load_state_dict(sd)
    return m.to(dev).eval().requires_grad_(False), cfg, f"{run.name}-{ck.get('step', ck.get('global_step'))}"


def logreg(x, y, iters=500, wd=1e-4):
    """Логистическая регрессия (L-BFGS) на стандартизованных признаках, классы взвешены по размеру:
    иначе 254 кадра Терры против 8000 других тонут, и модель всегда отвечает «не Терра». x : (N, d), y : (N,) из 0 и 1."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    w = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(1, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    pos = ((y == 0).sum() / (y == 1).sum()).float()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xs @ w + b, y.float(), pos_weight=pos) + wd * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda q: torch.sigmoid(((q - mu) / sd) @ w.detach() + b.detach())


def main():
    a = parse_args()
    dev = "cuda"
    out = Path(a.out)
    (out / "grids").mkdir(parents=True, exist_ok=True)
    root = Path(a.dataset)
    z = np.load(root / "dino_feats.npz")
    paths = list(z["paths"])
    color = F.normalize(torch.from_numpy(z["color"]).float(), dim=-1).to(dev)
    rows = {r["path"]: r for r in read_manifest(root / "manifest.jsonl")}
    hold = set(json.loads((root / "terra_holdout.json").read_text()))
    is_terra = torch.tensor([rows[p]["source"] == "terra" for p in paths], device=dev)
    is_hold = torch.tensor([p in hold for p in paths], device=dev)
    tr_terra, ho_terra = color[is_terra & ~is_hold], color[is_hold]
    g = torch.Generator(device=dev).manual_seed(0)
    others = color[~is_terra][torch.randperm(int((~is_terra).sum()), generator=g, device=dev)]
    tr_other, ho_other = others[:8000], others[8000:10000]

    clf = logreg(torch.cat([tr_terra, tr_other]), torch.cat([torch.ones(len(tr_terra)), torch.zeros(len(tr_other))]).to(dev))
    print(f"классификатор: отложенная Терра узнана {(clf(ho_terra) > 0.5).float().mean():.0%}, "
          f"другие кошки приняты за Терру {(clf(ho_other) > 0.5).float().mean():.1%}", flush=True)
    nn_hold = (ho_terra @ tr_terra.T).max(1).values
    thr = torch.quantile(nn_hold, 0.95).item()
    print(f"новизна: у отложенных кадров близость к ближайшему обучающему, медиана {nn_hold.median():.3f}, 95% {thr:.3f}", flush=True)
    all_terra = color[is_terra].double().cpu()

    def dino_feats(img):
        """Картинки (B, 3, S, S) в [-1, 1] -> нормированные признаки DINOv2, тем же путем, что у сэмплов."""
        x = F.interpolate((img + 1) / 2, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        with bf16():
            return F.normalize(dino(pixel_values=(x - mean) / std).pooler_output.float(), dim=-1)

    model, cfg, name = load_unet(a.ckpt, dev)
    w_n, k = cfg["cond_sizes"]
    terra_type = cfg["terra_type"]
    sde = VPSDE(pred=cfg["pred"])
    vae = AutoencoderKL.from_pretrained(cfg["vae"], local_files_only=True, low_cpu_mem_usage=False).to(dev).eval()
    dino = AutoModel.from_pretrained(DINO, local_files_only=True).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    weak = {}

    # проверка пути признаков: отложенные фото Терры в 512 через dino_feats должны быть так же близки к обучающим,
    # как их признаки из dino_feats.npz (там кроп сразу в 224)
    hrows = [rows[p] for p in sorted(hold)]
    with torch.no_grad():
        x = torch.stack([FrameCrops(hrows, root, size=512, crops=1, seed=0)[i][0][0] for i in range(len(hrows))]).to(dev)
        check = (dino_feats(x) @ tr_terra.T).max(1).values.median().item()
    print(f"проверка пути признаков: у отложенных в 512 близость {check:.3f} (из dino_feats.npz {nn_hold.median():.3f})", flush=True)

    f = open(out / "metrics.csv", "a", newline="")
    wr = csv.writer(f)
    if f.tell() == 0:
        wr.writerow(["модель", "вариант", "n", "похожа_на_терру", "почти_копии", "близость_медиана", "kid_dinov2_x1000", "fd_dinov2", "секунд"])
    wr.writerow(["", "отложенные кадры Терры", len(ho_terra), f"{(clf(ho_terra) > 0.5).float().mean():.3f}",
                 f"{(nn_hold > thr).float().mean():.3f}", f"{nn_hold.median():.3f}", "", "", ""])
    f.flush()

    for spec in a.rows:
        opt = dict(kv.split("=", 1) for kv in spec.split(","))
        w = float(opt.get("w", 1))
        t0 = time.time()
        feats = []
        for b in range(0, a.n, a.batch):
            n = min(a.batch, a.n - b)
            s = a.seed * 1_000_003 + b
            x0 = torch.randn(n, 4, 64, 64, generator=torch.Generator().manual_seed(s)).to(dev)
            t = opt.get("type", "none")
            scene = torch.full((n,), k if t == "none" else terra_type if t == "terra" else int(t), device=dev)
            y = torch.stack([torch.ones(n, dtype=torch.long, device=dev), scene], 1)
            y0 = torch.tensor([w_n, k], device=dev)
            net = CFG(model, y, y0, w)
            if "ag" in opt:
                path, wa = opt["ag"].split("@")
                if path not in weak:
                    weak[path] = load_unet(path, dev)[0]
                net = Guided(net, CFG(weak[path], y, y0, w), float(wa))
            torch.manual_seed(s)
            with torch.no_grad():
                lat = sample(net, sde, x0, steps=a.steps, rho=1.0, method="sde", amp_ctx=bf16, clip=False)
                img = torch.cat([vae.decode(lat[j:j + 16].float() / cfg["scale"]).sample for j in range(0, n, 16)]).clamp(-1, 1)
                if b == 0:
                    save_image((img[:32] + 1) / 2, out / "grids" / f"{name}_{spec.replace(',', '_').replace('/', '-')}.jpg", nrow=8)
                feats.append(dino_feats(img))
        e = torch.cat(feats)
        nn_gen = (e @ tr_terra.T).max(1).values
        kd, _ = kid(all_terra, e.double().cpu())
        r = [name, spec, len(e), f"{(clf(e) > 0.5).float().mean():.3f}", f"{(nn_gen > thr).float().mean():.3f}",
             f"{nn_gen.median():.3f}", f"{kd * 1e3:.2f}", f"{fd(all_terra, e.double().cpu()):.1f}", f"{time.time() - t0:.0f}"]
        wr.writerow(r)
        f.flush()
        print(f"{name} {spec}: похожа на Терру {r[3]}, почти копии {r[4]}, близость {r[5]}, KID-DINOv2 x1000 {r[6]}, FD {r[7]}, {r[8]} с",
              flush=True)


if __name__ == "__main__":
    main()
