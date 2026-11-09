"""
Guidance during generation. The wrappers behave like a model: forward(x, c) -> network output,
so the sampler sample() from train.py does not know about them. Wrappers can be nested.

  CFG     : classifier-free guidance. One model with conditions y, the output is pushed away from the response with "empty"
            or weakened conditions y0:  out = D(y0) + w (D(y) - D(y0)).
  Guided  : autoguidance (Karras et al. 2024). A weak model of the same kind:  out = bad + w (good - bad).

The output is linear in the network responses, and the conversion of v to noise is linear for fixed x and t, so the outputs can be mixed directly.
"""
import torch
import torch.nn as nn


class CFG(nn.Module):
    """y, y0 : (number of conditions,) or (B, number of conditions) long. The value size of a condition = "empty"."""

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
        out = self.model(torch.cat([x, x]), torch.cat([c, c]), torch.cat([y, y0]))   # one call for both responses
        cond, uncond = out.chunk(2)
        return uncond + self.w * (cond - uncond)


class Guided(nn.Module):
    """Autoguidance. good and bad are called as models without conditions: wrap them in CFG if conditions are needed."""

    def __init__(self, good, bad, w):
        super().__init__()
        self.good, self.bad, self.w = good, bad, w

    def forward(self, x, c):
        g, b = self.good(x, c), self.bad(x, c)
        return b + self.w * (g - b)
