"""Instance normalization of half the feature channels."""

import torch
from torch import nn


class HalfInstanceNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        assert width % 2 == 0
        self.half = width // 2
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones(1, self.half, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.half, 1, 1))

    def forward(self, x):
        a, b = x.split(self.half, 1)
        with torch.autocast(x.device.type, enabled=False):
            value = a.float()
            mu = value.mean((2, 3), keepdim=True)
            var = (value - mu).square().mean((2, 3), keepdim=True)
            a = (value - mu) * torch.rsqrt(var + self.eps) * self.weight.float() + self.bias.float()
        return torch.cat((a.to(x.dtype), b), 1)
