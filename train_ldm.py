#!/usr/bin/env python
"""
Латентная диффузия на Lightning, несколько GPU одного узла. Модель, расписание шума и сэмплер из train.py.

Запуск: train_ldm.sbatch. Запуск с той же --out продолжается с last.ckpt.
В --out: metrics.csv (eps-MSE как в train.py, сравним между запусками, норма градиента до обрезки, худшее внимание),
attn.csv (максимум логитов и энтропия каждого блока внимания), samples/, ckpt/last.ckpt, ema_*.pt,
raw/raw_*.pt (сырые веса в bf16, чтобы разбирать срывы обучения).
"""
import argparse
import copy
import csv
import json
import math
import platform
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
from models.unet import AttnBlock
from train import VPSDE, cosine_lr, diffusion_loss, ema_update, sample, save_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="папка датасета с latents/")
    p.add_argument("--out", required=True)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--batch", type=int, default=64, help="на один GPU")
    p.add_argument("--workers", type=int, default=4, help="на один GPU")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=2000, help="шагов линейного прогрева LR, потом косинус")
    p.add_argument("--seed", type=int, default=0, help="у процесса номер r сид seed * 1000 + r")
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pred", default="v", choices=["eps", "v"])
    p.add_argument("--base", type=int, default=192)
    p.add_argument("--mults", default="1,2,3,4")
    p.add_argument("--num-res", type=int, default=2)
    p.add_argument("--attn-res", default="32,16,8")
    p.add_argument("--head-ch", type=int, default=64, help="каналов на голову внимания, 0 = одна голова")
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=5000)
    p.add_argument("--raw-every", type=int, default=1000, help="сырые веса в raw/, 0 = не сохранять")
    p.add_argument("--attn-every", type=int, default=100, help="статистика внимания на 8 латентах батча")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--sample-method", default="sde", choices=["sde", "heun"],
                   help="сэмплер сеток по ходу обучения: Heun 50 у ldm_v2 давал размазанные картинки, SDE 250 честнее")
    p.add_argument("--sample-steps", type=int, default=250)
    p.add_argument("--final-steps", type=int, default=1000, help="шагов SDE на финальной сетке")
    p.add_argument("--limit", type=int, default=0, help="для проверки: кадров из датасета")
    return p.parse_args()


class LDM(L.LightningModule):
    def __init__(self, a, scale):
        super().__init__()
        self.a, self.scale = a, scale
        self.model = UNet(img_size=64, in_ch=4, base=a.base, mults=[int(v) for v in a.mults.split(",")], num_res=a.num_res,
                          attn_res=[int(v) for v in a.attn_res.split(",")], dropout=a.dropout, norm_groups=32,
                          head_ch=a.head_ch or None, qk_norm=a.qk_norm)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.sde = VPSDE(pred=a.pred)
        # в списке, чтобы Lightning не считал это отдельным подмодулем с другими именами весов
        self.net = [self.model if a.no_compile else torch.compile(self.model)]
        self.t_log, self.acc, self.gnorm = time.time(), [], []
        self.attn_last = (float("nan"), float("nan"))

    def configure_optimizers(self):
        a = self.a
        opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, a) / a.lr)
        return dict(optimizer=opt, lr_scheduler=dict(scheduler=sched, interval="step"))

    def on_train_start(self):
        # seed_everything дал всем процессам один сид, и t, шум и dropout совпадали на всех GPU.
        # Шаг в сиде: после продолжения с чекпоинта шум не повторяет начало обучения
        torch.manual_seed(self.a.seed * 1000 + self.global_rank + 1_000_003 * self.global_step)

    def on_fit_start(self):
        # какой физический GPU у процесса: чтобы сбой CUDA можно было привязать к карте
        pr = torch.cuda.get_device_properties(self.device)
        print(f"процесс {self.global_rank}: {platform.node()}, GPU {self.device}, pci {getattr(pr, 'pci_bus_id', '?')}, "
              f"uuid {getattr(pr, 'uuid', '?')}", flush=True)
        self.vae = None
        if self.trainer.is_global_zero:
            out = Path(self.a.out)
            (out / "samples").mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps({**vars(self.a), "scale": self.scale}, indent=2, ensure_ascii=False))
            new = not (out / "metrics.csv").exists()
            self.log_f = open(out / "metrics.csv", "a", newline="")
            self.log_w = csv.writer(self.log_f)
            if new:
                self.log_w.writerow(["step", "loss", "loss_t0-0.2", "loss_t0.2-0.4", "loss_t0.4-0.6", "loss_t0.6-0.8", "loss_t0.8-1", "lr", "it_per_s",
                                     "grad_norm", "grad_norm_max", "attn_max_logit", "attn_min_entropy"])
            self.attn_names = [n for n, m in self.model.named_modules() if isinstance(m, AttnBlock)]
            new = not (out / "attn.csv").exists()
            self.attn_f = open(out / "attn.csv", "a", newline="")
            self.attn_w = csv.writer(self.attn_f)
            if new:
                self.attn_w.writerow(["step"] + [f"{n}_{k}" for n in self.attn_names for k in ("max", "ent")])
            (out / "raw").mkdir(exist_ok=True)
            self.vae = [AutoencoderKL.from_pretrained(self.a.vae, local_files_only=True, low_cpu_mem_usage=False).to(self.device).eval().requires_grad_(False)]
            self.noise = torch.randn(self.a.n_samples, 4, 64, 64, generator=torch.Generator().manual_seed(1234)).to(self.device)

    def training_step(self, x0, idx):
        loss, t, per = diffusion_loss(self.net[0], self.sde, x0)
        if self.global_step == 0 and idx == 0:
            first = self.all_gather(t[:3].detach())          # (процессов, 3): строки должны различаться
            if self.trainer.is_global_zero:
                print("первые t на каждом процессе:\n" + "\n".join(" ".join(f"{v:.4f}" for v in r) for r in first.view(-1, 3).tolist()), flush=True)
        self.acc.append((t.detach(), per.detach()))
        return loss

    def on_before_optimizer_step(self, optimizer):
        # градиенты уже сложены между процессами, но еще не обрезаны
        if self.trainer.is_global_zero:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            self.gnorm.append(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).detach())

    @torch.no_grad()
    def attn_stats(self, x0):
        """Максимум логитов и средняя энтропия внимания (доля от log N) каждого блока на 8 латентах, t от 0.05 до 0.95."""
        x0 = x0[:8].float()
        n = len(x0)
        t = torch.linspace(0.05, 0.95, n, device=x0.device)
        eps = torch.randn(x0.shape, generator=torch.Generator(device=x0.device).manual_seed(0), device=x0.device)
        xt = self.sde.m(t).view(-1, 1, 1, 1) * x0 + self.sde.sigma(t).view(-1, 1, 1, 1) * eps
        res = {}

        def hook(name):
            def f(mod, inp, out):
                lg = mod.logits(inp[0])
                p = lg.softmax(-1)
                ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean() / math.log(lg.shape[-1])
                res[name] = (lg.max().item(), ent.item())
            return f

        mods = dict(self.model.named_modules())
        hs = [mods[n].register_forward_hook(hook(n)) for n in self.attn_names]
        self.model.eval()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.model(xt, self.sde.c(t))
        finally:
            self.model.train()
            for h in hs:
                h.remove()
        return [res[n] for n in self.attn_names]

    def on_train_batch_end(self, outputs, batch, idx):
        a, s = self.a, self.global_step
        ema_update(self.ema, self.model, min(a.ema, (1 + s) / (10 + s)))
        if not self.trainer.is_global_zero:
            self.acc.clear()
            return
        if a.attn_every and (s % a.attn_every == 0 or s == a.steps):
            st = self.attn_stats(batch)
            self.attn_w.writerow([s] + [f"{v:.4g}" for pair in st for v in pair])
            self.attn_f.flush()
            self.attn_last = (max(m for m, _ in st), min(e for _, e in st))
        if s % a.log_every == 0 or s == a.steps:
            t = torch.cat([v[0] for v in self.acc]); per = torch.cat([v[1] for v in self.acc])
            b = torch.bucketize(t, torch.linspace(0, 1, 6, device=t.device)[1:-1])
            bins = [per[b == i].mean().item() if (b == i).any() else float("nan") for i in range(5)]
            g = torch.stack(self.gnorm) if self.gnorm else torch.full((1,), float("nan"))
            ips = len(self.acc) / (time.time() - self.t_log)
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            am, ae = self.attn_last
            self.log_w.writerow([s, f"{per.mean().item():.5f}"] + [f"{v:.5f}" for v in bins] + [f"{lr:.2e}", f"{ips:.2f}",
                                f"{g.mean().item():.4g}", f"{g.max().item():.4g}", f"{am:.4g}", f"{ae:.4f}"])
            self.log_f.flush()
            eta = (a.steps - s) / max(ips, 1e-9) / 3600
            print(f"step {s:7d} | loss {per.mean().item():.4f} | по t: " + " ".join(f"{v:.3f}" for v in bins)
                  + f" | grad {g.mean().item():.3f} (max {g.max().item():.3f}) | attn max {am:.1f}, ent min {ae:.2f}"
                  + f" | lr {lr:.1e} | {ips:.2f} it/s | осталось ~{eta:.1f} ч", flush=True)
            self.acc.clear()
            self.gnorm.clear()
            self.t_log = time.time()
        if s % a.sample_every == 0 or s == a.steps:
            self.make_samples(Path(a.out) / "samples" / f"step_{s:07d}.png")
            self.t_log = time.time()
        if a.snapshot_every and s % a.snapshot_every == 0:
            torch.save(dict(ema=self.ema.state_dict(), step=s, args=vars(a), scale=self.scale), Path(a.out) / f"ema_{s:07d}.pt")
        if a.raw_every and s % a.raw_every == 0:
            raw = {k: v.to(torch.bfloat16) for k, v in self.model.state_dict().items()}
            torch.save(dict(model=raw, step=s, args=vars(a), scale=self.scale), Path(a.out) / "raw" / f"raw_{s:07d}.pt")

    @torch.no_grad()
    def make_samples(self, path, method=None, steps=None):
        t0 = time.time()
        method = method or self.a.sample_method
        bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        # шум SDE одинаковый на всех сетках; fork_rng возвращает генератор обучения в прежнее состояние
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(1234)
            z = sample(self.ema, self.sde, self.noise, steps=steps or self.a.sample_steps, rho=1.0 if method == "sde" else 2.0,
                       method=method, amp_ctx=bf16, clip=False)
        with torch.autocast("cuda", enabled=False):   # Lightning держит bf16 снаружи, VAE считаем в float32
            x = torch.cat([self.vae[0].decode(z[i:i + 4].float() / self.scale).sample for i in range(0, len(z), 4)])
        save_grid(x, path, nrow=4, upscale=1)
        print(f"  сэмплы {path.name}: {time.time() - t0:.0f} с, std латента {z.std().item():.2f}", flush=True)

    def on_train_end(self):
        if self.trainer.is_global_zero:
            self.make_samples(Path(self.a.out) / "samples" / f"final_sde{self.a.final_steps}.png", method="sde", steps=self.a.final_steps)


def lr_at(step, a):
    """Линейный прогрев до a.lr за a.warmup шагов, затем косинус до a.lr_min к шагу a.steps."""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    return cosine_lr(step - a.warmup, a.steps - a.warmup, a.lr, a.lr_min)


def main():
    a = parse_args()
    L.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    ds = CatLatents(a.data, a.limit)
    print(f"кадров {len(ds)} (нет в манифесте {ds.dropped}), кропов на кадр {ds.z.shape[1]}, scale {ds.scale:.5f}", flush=True)
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
