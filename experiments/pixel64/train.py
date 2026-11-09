#!/usr/bin/env python
"""
DDPM-style diffusion model (VP-SDE, noise or velocity v prediction) on cat photos.

  forward process  x_t = m(t) x_0 + sigma(t) eps,  beta(t) schedule linear or cosine (--schedule)
  loss             || eps_theta(x_t, t) - eps ||^2 or the same for v = m eps - sigma x_0 (--pred),
                   Min-SNR weights (--min-snr),  t ~ U(t_min, 1)
  generation       reverse ODE (Euler / Heun) or reverse SDE (Euler-Maruyama)

The architecture is chosen by name (--model), its hyperparameters are set in terra_diffusion/models/__init__.py.

Example:
  python experiments/pixel64/train.py --data-dir dataset_unique --out-dir runs/cat64 --model unet --resume
Generation only, from a checkpoint:
  python experiments/pixel64/train.py --out-dir runs/cat64 --sample-only --sample-method sde --sample-steps 1000
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root: terra_diffusion imports without installing
from terra_diffusion.diffusion import cosine_lr, diffusion_loss, ema_update, sample, sde_from_cfg
from terra_diffusion.images import center_crops, crop_offsets, load_images, min_crop_side, save_grid, to_unit
from terra_diffusion.models import CONFIGS, build_model, config_from_checkpoint


# Extra augmentations, applied after the crop and flip to the uint8 image 3 x S x S.
# Edited by hand. Example against memorization:
#   EXTRA_AUG = [v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)]
EXTRA_AUG = []


# ----------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description="DDPM / VP-SDE on cat photos")
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", type=str, default="dataset_unique")
    g.add_argument("--exclude-file", type=str, default=None, help="file with names of photos to exclude")
    g.add_argument("--img-size", type=int, default=64, help="resolution to train at")
    g.add_argument("--pre-size", type=int, default=128, help="short side after the initial downscale")
    g.add_argument("--min-scale", type=float, default=0.85, help="minimum crop side as a fraction of the short side")
    g.add_argument("--boxes", type=str, default=None,
                   help="boxes.json: the crop always contains the whole cat box, --min-scale is then unused")
    g.add_argument("--workers", type=int, default=8)

    g = p.add_argument_group("model")
    g.add_argument("--model", type=str, default="unet", choices=sorted(CONFIGS),
                   help="preset from terra_diffusion/models/__init__.py")

    g = p.add_argument_group("diffusion")
    g.add_argument("--beta-min", type=float, default=0.1)
    g.add_argument("--beta-max", type=float, default=20.0)
    g.add_argument("--t-min", type=float, default=1e-3)
    g.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"],
                   help="noise schedule: linear beta (DDPM) or cosine (Improved DDPM)")
    g.add_argument("--cond", type=str, default="log_snr", choices=["log_snr", "log_sigma"],
                   help="network input: log(sigma/m) or log sigma (old runs)")
    g.add_argument("--pred", type=str, default="eps", choices=["eps", "v"],
                   help="what the network predicts: noise eps or velocity v = m eps - sigma x_0")
    g.add_argument("--min-snr", type=float, default=0.0,
                   help="gamma for Min-SNR weights (Hang et al. 2023, usually 5), 0 = no weights")

    g = p.add_argument_group("training")
    g.add_argument("--out-dir", type=str, default="runs/cat64")
    g.add_argument("--steps", type=int, default=100_000)
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--lr", type=float, default=2e-4, help="lr at the start of cosine annealing")
    g.add_argument("--lr-min", type=float, default=1e-6, help="lr at the end of cosine annealing")
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--ema", type=float, default=0.9995)
    g.add_argument("--amp", type=str, default="auto", choices=["auto", "fp16", "bf16", "none"])
    g.add_argument("--no-compile", action="store_true", help="do not use torch.compile")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--resume", action="store_true", help="continue from out-dir/last.pt if it exists")

    g = p.add_argument_group("logging and generation")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--sample-every", type=int, default=5000)
    g.add_argument("--ckpt-every", type=int, default=5000)
    g.add_argument("--snapshot-every", type=int, default=20000, help="separate EMA snapshots for comparison")
    g.add_argument("--n-samples", type=int, default=64)
    g.add_argument("--sample-steps", type=int, default=1000)
    g.add_argument("--sample-method", type=str, default="sde", choices=["euler", "heun", "sde"])
    g.add_argument("--rho", type=float, default=2.0, help="step clustering near t = 0 for the ODE (1 is used for the SDE)")
    g.add_argument("--sample-only", action="store_true")
    g.add_argument("--ckpt", type=str, default=None, help="checkpoint for --sample-only (default out-dir/last.pt)")
    return p.parse_args()


# ----------------------------------------------------------------------------- data


def load_boxes(path, files, imgs):
    """Boxes from boxes.json in pixels of the downscaled photos, rounded outward. Same order as files."""
    data = json.loads(Path(path).read_text())
    out = []
    for p, img in zip(files, imgs):
        if p.name not in data:
            raise SystemExit(f"{path} has no box for {p.name}")
        r = data[p.name]
        b = r["box"]
        _, H, W = img.shape
        kx, ky = W / r["width"], H / r["height"]
        out.append((max(math.floor(b["x1"] * kx), 0), max(math.floor(b["y1"] * ky), 0),
                    min(math.ceil(b["x2"] * kx), W), min(math.ceil(b["y2"] * ky), H)))
    return out


class CatDataset(Dataset):
    """
    Random square crop + downscale to img_size + random flip (+ EXTRA_AUG).
    With boxes the crop always contains the cat box: the side is random from the smallest allowed up to the short side of the photo,
    the offset is random within the range where the box stays inside.
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
            # square with side from min_scale to 1 of the short side, at a random location
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


# ----------------------------------------------------------------------------- diffusion


# ----------------------------------------------------------------------------- utilities


@torch.no_grad()
def nn_check(samples, ref_all, n_ref):
    """For each sample the nearest training image (accounting for flips) and the pixel distance."""
    d = torch.cdist(samples.flatten(1).float(), ref_all.flatten(1).float())
    dmin, idx = d.min(dim=1)
    return dmin, idx % n_ref


@torch.no_grad()
def train_nn_baseline(ref, ref_all):
    """Typical distance from a training image to the nearest other one (not itself or its own flip)."""
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
    pairs = torch.stack([x[:k], ref[idx[:k]]], dim=1).flatten(0, 1)  # sample, nearest training image, sample, ...
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

    # ---- data
    exclude = set()
    if args.exclude_file:
        for line in Path(args.exclude_file).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                exclude.add(line)
    t0 = time.time()
    files, imgs = load_images(args.data_dir, args.pre_size, exclude)
    n_port = sum(im.shape[1] > im.shape[2] for im in imgs)
    print(f"photos: {len(imgs)}, portrait {n_port}, landscape {len(imgs) - n_port}, "
          f"loading took {time.time() - t0:.1f} s", flush=True)
    if len(imgs) == 0:
        raise SystemExit(f"no photos found in {args.data_dir}")

    boxes = load_boxes(args.boxes, files, imgs) if args.boxes else None
    ds = CatDataset(imgs, args.img_size, args.min_scale, boxes=boxes)
    save_grid(torch.stack([ds[i * 7 % len(ds)] for i in range(64)]), out / "data_preview.png")

    ref = center_crops(imgs, args.img_size).to(device)
    ref_all = torch.cat([ref, ref.flip(-1)])
    base_nn = train_nn_baseline(ref, ref_all)
    print(f"distance between nearest training images: median {base_nn.median():.2f}, "
          f"10% quantile {base_nn.quantile(0.1):.2f}", flush=True)

    # ---- model
    sde = sde_from_cfg(vars(args))
    ckpt_path = Path(args.ckpt) if args.ckpt else out / "last.pt"
    ck = None
    if (args.resume or args.sample_only) and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    elif args.sample_only:
        raise SystemExit(f"no checkpoint {ckpt_path}")

    model, model_cfg = build_model(args.model, args.img_size)
    if ck is not None:
        for key, old in (("cond", "log_sigma"), ("schedule", "linear"), ("pred", "eps")):
            ck_val = ck["args"].get(key, old)
            if ck_val != getattr(args, key):
                raise SystemExit(f"checkpoint was trained with --{key} {ck_val}, but now --{key} {getattr(args, key)}")
        old_cfg = config_from_checkpoint(ck)
        if old_cfg != model_cfg:
            raise SystemExit(f"model configuration in the checkpoint differs from --model {args.model}:\n"
                             f"  in checkpoint {old_cfg}\n  now           {model_cfg}")
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model {args.model}: {n_par / 1e6:.1f} M parameters | {model_cfg}", flush=True)

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
        print(f"loaded checkpoint {ckpt_path}, step {start_step}", flush=True)
        del ck

    # the same starting noise for all checks
    fixed_noise = torch.randn(args.n_samples, 3, args.img_size, args.img_size,
                              generator=torch.Generator().manual_seed(1234)).to(device)

    if args.sample_only:
        prefix = out / f"final_{args.sample_method}_{args.sample_steps}_step{start_step}"
        _, dmin, dt = run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all, prefix)
        print(f"done in {dt:.1f} s: {prefix}.png | NN distance median {dmin.median():.2f} "
              f"(training images {base_nn.median():.2f})", flush=True)
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
        print("torch.compile enabled, the first steps will be slow while compiling", flush=True)

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

        # statistics: the logs get the unweighted eps-MSE, it is comparable between runs
        loss_sum += per.detach().mean()
        b = torch.bucketize(t, edges)
        bin_sum.index_add_(0, b, per.detach())
        bin_cnt.index_add_(0, b, torch.ones_like(per))
        g = float(gnorm)
        if math.isfinite(g):  # in fp16 GradScaler sometimes skips a step with inf, we do not count it
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
            print(f"step {s:7d} | loss {l:.4f} | by t: " + " ".join(f"{v:.3f}" for v in bins)
                  + f" | lr {lr:.1e} | grad {gnorm_sum / n_acc:.2f} | {ips:.1f} it/s | ~{eta:.1f} h left",
                  flush=True)
            loss_sum.zero_(); bin_sum.zero_(); bin_cnt.zero_()
            gnorm_sum, n_acc, t_log = 0.0, 0, time.time()

        if s % args.sample_every == 0 or s == args.steps:
            _, dmin, dt = run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all,
                                       out / "samples" / f"step_{s:07d}")
            nn_w.writerow([s, f"{dmin.median():.3f}", f"{dmin.min():.3f}", f"{base_nn.median():.3f}", f"{dt:.1f}"])
            nn_f.flush()
            print(f"  samples saved ({dt:.1f} s) | NN distance median {dmin.median():.2f}, "
                  f"min {dmin.min():.2f} (training images {base_nn.median():.2f})", flush=True)
            t_log = time.time()  # generation time is not included in it/s

        if s % args.ckpt_every == 0 or s == args.steps:
            atomic_save(checkpoint(s), out / "last.pt")
        if args.snapshot_every and s % args.snapshot_every == 0:
            torch.save({"ema": ema.state_dict(), "step": s, "args": vars(args),
                        "model_name": args.model, "model_cfg": model_cfg}, out / f"ema_{s:07d}.pt")

    # final images: deterministic ODE and stochastic SDE
    if start_step < args.steps:
        run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all, out / "final_heun50", "heun", 50)
        run_sampling(ema, sde, fixed_noise, args, amp_ctx, ref, ref_all, out / "final_sde1000", "sde", 1000)
    print(f"training finished in {(time.time() - t_start) / 3600:.2f} h", flush=True)
    log_f.close()
    nn_f.close()


if __name__ == "__main__":
    main()
