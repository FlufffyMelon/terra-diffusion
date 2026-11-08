"""
Направление (guidance) при генерации. Обертки ведут себя как модель: forward(x, c) -> выход сети,
поэтому сэмплер sample() из train.py о них не знает. Обертки можно вкладывать друг в друга.

  CFG     : classifier-free guidance. Одна модель с условиями y, выход отодвигается от ответа с «пустыми»
            или ослабленными условиями y0:  out = D(y0) + w (D(y) - D(y0)).
  Guided  : autoguidance (Karras et al. 2024). Слабая модель той же природы:  out = bad + w (good - bad).

Выход линеен по ответам сети, а перевод v в шум линеен при фиксированных x и t, поэтому смешивать можно прямо выходы.
"""
import torch
import torch.nn as nn


class CFG(nn.Module):
    """y, y0 : (число условий,) или (B, число условий) long. Значение size у условия = «пусто»."""

    def __init__(self, model, y, y0, w):
        super().__init__()
        self.model, self.w = model, w
        self.register_buffer("y", torch.as_tensor(y), persistent=False)
        self.register_buffer("y0", torch.as_tensor(y0), persistent=False)

    def forward(self, x, c):
        y = self.y.to(x.device).expand(len(x), -1) if self.y.dim() == 1 else self.y.to(x.device)
        y0 = self.y0.to(x.device).expand(len(x), -1) if self.y0.dim() == 1 else self.y0.to(x.device)
        if self.w == 1:
            return self.model(x, c, y)
        out = self.model(torch.cat([x, x]), torch.cat([c, c]), torch.cat([y, y0]))   # один вызов на оба ответа
        cond, uncond = out.chunk(2)
        return uncond + self.w * (cond - uncond)


class Guided(nn.Module):
    """Autoguidance. good и bad вызываются как модели без условий: оберните их в CFG, если условия нужны."""

    def __init__(self, good, bad, w):
        super().__init__()
        self.good, self.bad, self.w = good, bad, w

    def forward(self, x, c):
        g, b = self.good(x, c), self.bad(x, c)
        return b + self.w * (g - b)
