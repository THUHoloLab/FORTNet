"""Calibrated exposure response and deterministic sampling."""

import numpy as np
import torch

from .assets import load_response

SEED = 20260909
from .residual import DarkResidual


class CalibratedExposure:
    def __init__(self, device):
        spec = load_response()
        self.residual = DarkResidual(device)
        values = torch.tensor(spec["curves"], dtype=torch.float32, device=device)
        identity = torch.arange(256, device=device, dtype=torch.float32).div(255).expand(1, 3, 256)
        self.curves = torch.cat((values[:2], identity, values[2:]), 0)
        self.log_anchors = torch.tensor([0.25, 0.5, 1.0, 1.5, 2.0], device=device).log()

    def apply(self, raw, gains, update):
        if type(update) is not int or update < 1:
            raise ValueError("Explicit positive update seed required")
        gains = torch.as_tensor(gains, device=raw.device, dtype=torch.float32)
        if (
            gains.shape != (len(raw),)
            or not torch.isfinite(gains).all()
            or (gains < 0.25).any()
            or (gains > 2).any()
        ):
            raise ValueError("Finite per-image exposure0.25..2 required")
        logs = gains.log()
        high = torch.bucketize(logs, self.log_anchors, right=True).clamp(1, 4)
        low = high - 1
        alpha = (
            (logs - self.log_anchors[low]) / (self.log_anchors[high] - self.log_anchors[low])
        ).reshape(-1, 1, 1)
        tables = self.curves[low] * (1 - alpha) + self.curves[high] * alpha
        indices = (raw.detach() * 255).round().clamp(0, 255).long().flatten(2)
        output = torch.gather(tables, 2, indices).reshape_as(raw)
        output = (output * 255).round().clamp(0, 255) / 255
        output = torch.where((gains == 1).reshape(-1, 1, 1, 1), raw, output)
        dark = (gains == 0.25) | (gains == 0.5)
        if dark.any():
            output[dark] = self.residual.apply(output[dark], gains[dark], update)
        return output


def draws(update, batch):
    if type(update) is not int or update < 1:
        raise ValueError("Positive global update required")
    rng = np.random.default_rng([SEED, 10, update])
    anchors = np.array([0.25, 0.5, 1.5, 2.0])
    choose = rng.random(batch) < 0.5
    fixed = anchors[rng.integers(0, 4, batch)]
    continuous = np.exp(rng.uniform(np.log(0.25), np.log(2.0), batch))
    return np.where(choose, fixed, continuous).astype(np.float32)
