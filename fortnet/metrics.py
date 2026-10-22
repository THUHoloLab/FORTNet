"""Per-image float RGB reconstruction metrics."""

import numpy as np
import torch
from torch.nn import functional as F


def metrics(pred, target):
    pred, target = (pred.float().clamp(0, 1), target.float())
    mse = (pred - target).square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    rgb = (pred.mean((2, 3)) - target.mean((2, 3))).abs().mean(1)
    coeff = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    y, g = ((pred * coeff).sum(1), (target * coeff).sum(1))
    bias = (y - g).mean((1, 2))
    edge = 0.5 * (
        (y.diff(dim=1) - g.diff(dim=1)).abs().mean((1, 2))
        + (y.diff(dim=2) - g.diff(dim=2)).abs().mean((1, 2))
    )
    p, t = (pred.double(), target.double())
    coord = torch.arange(11, device=p.device, dtype=p.dtype) - 5
    kernel = torch.exp(-coord.square() / (2 * 1.5**2))
    kernel /= kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11)
    conv = lambda x: F.conv2d(x, kernel, groups=3)
    mp, mt = (conv(p), conv(t))
    sp, st, cross = (conv(p * p) - mp * mp, conv(t * t) - mt * mt, conv(p * t) - mp * mt)
    ssim = (
        (
            (2 * mp * mt + 0.01**2)
            * (2 * cross + 0.03**2)
            / ((mp * mp + mt * mt + 0.01**2) * (sp + st + 0.03**2))
        )
        .flatten(1)
        .mean(1)
    )
    return {
        k: v.detach().cpu().tolist()
        for k, v in dict(
            psnr=psnr,
            ssim=ssim,
            rgb_mean_error=rgb,
            luma_bias=bias,
            brightness_error=bias.abs(),
            edge_error=edge,
        ).items()
    }


def summarize(rows):
    result = {}
    for key in ("psnr", "ssim", "rgb_mean_error", "luma_bias", "brightness_error", "edge_error"):
        a = np.asarray([r[key] for r in rows])
        result[key] = dict(
            mean=float(a.mean()),
            p05=float(np.quantile(a, 0.05)),
            p50=float(np.quantile(a, 0.5)),
            p95=float(np.quantile(a, 0.95)),
        )
    result["worst_decile_mean_psnr"] = float(
        np.mean(sorted((r["psnr"] for r in rows))[: max(1, len(rows) // 10)])
    )
    return result
