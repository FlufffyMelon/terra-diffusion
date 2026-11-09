"""
Diffusion: the VP-SDE forward process, the training loss and the samplers (Euler and Heun ODE, Euler-Maruyama SDE).
"""
import math
from contextlib import nullcontext

import numpy as np
import torch

class VPSDE:
    """
    Forward process x_t = m(t) x_0 + sigma(t) eps,  m = exp(-1/2 int_0^t beta),  sigma^2 = 1 - m^2.

    schedule = how the noise grows:
      linear : beta(t) = beta_min + t (beta_max - beta_min), as in DDPM.
      cosine : m(t) = cos(theta(t)) / cos(theta_0), as in Improved DDPM (Nichol, Dhariwal 2021).
               theta goes linearly from theta_0 = pi/2 * s/(1+s) to the angle where m = m_end.
               m_end is taken as for linear at t = 1, so the final noise is the same.
               The noise grows more smoothly, and mid noise levels get more training.
    cond = what the network receives as the "noise level":
      log_sigma : c = log sigma(t). The old variant, barely changes for t > 0.5.
      log_snr   : c = log(sigma / m) = -1/2 log SNR. Grows over the whole range.
    pred = what the network predicts:
      eps : noise eps, as in DDPM.
      v   : velocity v = m eps - sigma x_0 (Salimans, Ho 2022). At low noise it is almost eps, at high noise almost -x_0.
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
        # computed in float64: for small t int_beta is about 1e-5, and float32 loses precision
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
        return torch.sqrt(-torch.expm1(-self.int_beta(t)))  # sqrt(1 - m^2), accurate for small t too

    def c(self, t):
        """Network input encoding the time t."""
        if self.cond == "log_sigma":
            return torch.log(self.sigma(t))
        return 0.5 * torch.log(torch.expm1(self.int_beta(t)))  # log(sigma / m), accurate for small t too

    def to_eps(self, out, xt, t):
        """Network output -> predicted noise. t : (B,). For v: eps = m v + sigma x_t, since m^2 + sigma^2 = 1."""
        if self.pred == "eps":
            return out
        return self.m(t).view(-1, 1, 1, 1) * out + self.sigma(t).view(-1, 1, 1, 1) * xt


def sde_from_cfg(cfg):
    """VPSDE from the checkpoint arguments. Old checkpoints: log_sigma, linear and eps."""
    return VPSDE(cfg["beta_min"], cfg["beta_max"], cfg["t_min"],
                 cfg.get("cond", "log_sigma"), cfg.get("schedule", "linear"), pred=cfg.get("pred", "eps"))


def diffusion_loss(net, sde, x0, min_snr=0.0):
    """
    Algorithm 1 from DDPM: add noise by the formula and learn to predict the noise or the velocity v.
    Min-SNR: the x_0 error weight is min(SNR, gamma); converted to the eps or v loss, the weights are as below.
    Returns the training loss, t and the eps-MSE of each image. The eps-MSE is for logging only: it is unweighted
    and on the same scale for any --pred (v error = eps error / m), so runs can be compared.
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
def sample(net, sde, x_start, steps=50, rho=2.0, method="heun", amp_ctx=nullcontext, clip=True):
    """Reverse process from t = 1 to t_min, then the final denoising x0 = (x - sigma eps) / m.
    clip clamps the result to [-1, 1], it is not needed for latents."""
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
        # right-hand side when moving backward: x <- x + h * drift;  k = 1 for the ODE, k = 2 for the SDE
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
    return x0.clamp(-1, 1) if clip else x0


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p.detach(), 1.0 - decay)


def cosine_lr(step, total, lr, lr_min):
    """Cosine annealing without warmup: lr at step 0, lr_min at step total."""
    p = min(step, total) / max(total, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * p))
