#!/usr/bin/env python
"""
Латентная диффузия на Lightning, несколько GPU одного узла. Модель, расписание шума и сэмплер из train.py.

Запуск: train_ldm.sbatch. Запуск с той же --out продолжается с last.ckpt.
В --out: metrics.csv (eps-MSE как в train.py, сравним между запусками), samples/, ckpt/last.ckpt, ema_*.pt.
"""
import argparse
import copy
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import lightning as L
import torch
from diffusers import AutoencoderKL
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from dataset import CatLatents
from models import UNet
from train import VPSDE, cosine_lr, diffusion_loss, ema_update, sample, save_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="папка датасета с latents/")
    p.add_argument("--out", required=True)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--batch", type=int, default=64, help="на один GPU")
    p.add_argument("--workers", type=int, default=4, help="на один GPU")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pred", default="v", choices=["eps", "v"])
    p.add_argument("--base", type=int, default=192)
    p.add_argument("--mults", default="1,2,3,4")
    p.add_argument("--num-res", type=int, default=2)
    p.add_argument("--attn-res", default="32,16,8")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=5000)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--final-steps", type=int, default=1000, help="шагов SDE на финальной сетке")
    p.add_argument("--limit", type=int, default=0, help="для проверки: кадров из датасета")
    return p.parse_args()


class LDM(L.LightningModule):
    def __init__(self, a, scale):
        super().__init__()
        self.a, self.scale = a, scale
        self.model = UNet(img_size=64, in_ch=4, base=a.base, mults=[int(v) for v in a.mults.split(",")], num_res=a.num_res,
                          attn_res=[int(v) for v in a.attn_res.split(",")], dropout=a.dropout, norm_groups=32)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.sde = VPSDE(pred=a.pred)
        # в списке, чтобы Lightning не считал это отдельным подмодулем с другими именами весов
        self.net = [self.model if a.no_compile else torch.compile(self.model)]
        self.t_log, self.acc = time.time(), []

    def configure_optimizers(self):
        a = self.a
        opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, a.steps, a.lr, a.lr_min) / a.lr)
        return dict(optimizer=opt, lr_scheduler=dict(scheduler=sched, interval="step"))

    def on_fit_start(self):
        self.vae = None
        if self.trainer.is_global_zero:
            out = Path(self.a.out)
            (out / "samples").mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps({**vars(self.a), "scale": self.scale}, indent=2, ensure_ascii=False))
            new = not (out / "metrics.csv").exists()
            self.log_f = open(out / "metrics.csv", "a", newline="")
            self.log_w = csv.writer(self.log_f)
            if new:
                self.log_w.writerow(["step", "loss", "loss_t0-0.2", "loss_t0.2-0.4", "loss_t0.4-0.6", "loss_t0.6-0.8", "loss_t0.8-1", "lr", "it_per_s"])
            self.vae = [AutoencoderKL.from_pretrained(self.a.vae, local_files_only=True, low_cpu_mem_usage=False).to(self.device).eval().requires_grad_(False)]
            self.noise = torch.randn(self.a.n_samples, 4, 64, 64, generator=torch.Generator().manual_seed(1234)).to(self.device)

    def training_step(self, x0, idx):
        loss, t, per = diffusion_loss(self.net[0], self.sde, x0)
        self.acc.append((t.detach(), per.detach()))
        return loss

    def on_train_batch_end(self, outputs, batch, idx):
        a, s = self.a, self.global_step
        ema_update(self.ema, self.model, min(a.ema, (1 + s) / (10 + s)))
        if not self.trainer.is_global_zero:
            self.acc.clear()
            return
        if s % a.log_every == 0 or s == a.steps:
            t = torch.cat([v[0] for v in self.acc]); per = torch.cat([v[1] for v in self.acc])
            b = torch.bucketize(t, torch.linspace(0, 1, 6, device=t.device)[1:-1])
            bins = [per[b == i].mean().item() if (b == i).any() else float("nan") for i in range(5)]
            ips = len(self.acc) / (time.time() - self.t_log)
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log_w.writerow([s, f"{per.mean().item():.5f}"] + [f"{v:.5f}" for v in bins] + [f"{lr:.2e}", f"{ips:.2f}"])
            self.log_f.flush()
            eta = (a.steps - s) / max(ips, 1e-9) / 3600
            print(f"step {s:7d} | loss {per.mean().item():.4f} | по t: " + " ".join(f"{v:.3f}" for v in bins)
                  + f" | lr {lr:.1e} | {ips:.2f} it/s | осталось ~{eta:.1f} ч", flush=True)
            self.acc.clear()
            self.t_log = time.time()
        if s % a.sample_every == 0 or s == a.steps:
            self.make_samples(Path(a.out) / "samples" / f"step_{s:07d}.png")
            self.t_log = time.time()
        if a.snapshot_every and s % a.snapshot_every == 0:
            torch.save(dict(ema=self.ema.state_dict(), step=s, args=vars(a), scale=self.scale), Path(a.out) / f"ema_{s:07d}.pt")

    @torch.no_grad()
    def make_samples(self, path, method="heun", steps=None, rho=2.0):
        t0 = time.time()
        bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        z = sample(self.ema, self.sde, self.noise, steps=steps or self.a.sample_steps, rho=rho, method=method, amp_ctx=bf16, clip=False)
        with torch.autocast("cuda", enabled=False):   # Lightning держит bf16 снаружи, VAE считаем в float32
            x = torch.cat([self.vae[0].decode(z[i:i + 4].float() / self.scale).sample for i in range(0, len(z), 4)])
        save_grid(x, path, nrow=4, upscale=1)
        print(f"  сэмплы {path.name}: {time.time() - t0:.0f} с, std латента {z.std().item():.2f}", flush=True)

    def on_train_end(self):
        if self.trainer.is_global_zero:
            self.make_samples(Path(self.a.out) / "samples" / f"final_sde{self.a.final_steps}.png", method="sde", steps=self.a.final_steps, rho=1.0)


def main():
    a = parse_args()
    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    ds = CatLatents(a.data, a.limit)
    print(f"кадров {len(ds)}, кропов на кадр {ds.z.shape[1]}, scale {ds.scale:.5f}", flush=True)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True,
                        persistent_workers=a.workers > 0)
    module = LDM(a, ds.scale)
    print(f"U-Net: {sum(p.numel() for p in module.model.parameters()) / 1e6:.0f} M параметров", flush=True)
    ckpt = ModelCheckpoint(dirpath=Path(a.out) / "ckpt", filename="step", every_n_train_steps=a.ckpt_every, save_last=True, save_top_k=0)
    trainer = L.Trainer(accelerator="gpu", devices=a.devices, strategy=DDPStrategy(find_unused_parameters=False) if a.devices > 1 else "auto",
                        precision="bf16-mixed", max_steps=a.steps, gradient_clip_val=a.grad_clip, callbacks=[ckpt], logger=False,
                        enable_progress_bar=False, enable_model_summary=False)
    last = Path(a.out) / "ckpt" / "last.ckpt"
    trainer.fit(module, loader, ckpt_path=str(last) if last.exists() else None)


if __name__ == "__main__":
    main()
