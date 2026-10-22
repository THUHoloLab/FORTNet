"""Multiplicative local feature processing at two spatial scales."""

import torch
from torch import nn

from .half_instance import HalfInstanceNorm
from .window_field import PixelChannelNorm


class GatedLocalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = HalfInstanceNorm(width)
        self.norm2 = PixelChannelNorm(width)
        self.expand1 = nn.Conv2d(width, 2 * width, 1)
        self.local3 = nn.Conv2d(width, width, 3, padding=1, groups=width, padding_mode="reflect")
        self.local7 = nn.Conv2d(width, width, 7, padding=3, groups=width, padding_mode="reflect")
        self.project1 = nn.Conv2d(width, width, 1)
        self.expand2 = nn.Conv2d(width, 2 * width, 1)
        self.project2 = nn.Conv2d(width, width, 1)
        self.register_buffer("gate_operand_limit", torch.tensor(32.0))
        self.scale1 = nn.Parameter(torch.full((1, width, 1, 1), 0.1))
        self.scale2 = nn.Parameter(torch.full((1, width, 1, 1), 0.1))

    def forward(self, x):
        u, v = self.expand1(self.norm1(x)).chunk(2, 1)
        value = self.project1(self.local3(u).clamp(-32.0, 32.0) * self.local7(v).clamp(-32.0, 32.0))
        x = x + self.scale1.to(value.dtype) * value
        u, v = self.expand2(self.norm2(x)).chunk(2, 1)
        value = self.project2(u * v)
        return x + self.scale2.to(value.dtype) * value
