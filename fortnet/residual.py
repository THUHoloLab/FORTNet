"""Intensity-dependent correlated residuals for dark exposures."""

import numpy as np
import torch

from .assets import RESIDUAL


class DarkResidual:
    def __init__(self, device):
        with np.load(RESIDUAL) as a:
            self.pb = torch.as_tensor(a["profile_bins"].copy(), device=device, dtype=torch.long)
            self.tables = {
                c: {
                    k: torch.as_tensor(a[c + "_" + k].copy(), device=device, dtype=torch.float32)
                    for k in ("mean", "std", "filter", "source_cholesky")
                }
                for c in ("dark_025", "dark_050")
            }

    def apply(self, synthetic, gains, update, white=None):
        gains = torch.as_tensor(gains, device=synthetic.device)
        if (
            gains.shape != (len(synthetic),)
            or not ((gains == 0.25) | (gains == 0.5) | (gains == 1.5) | (gains == 2.0)).all()
        ):
            raise ValueError("Expected calibrated exposure anchors")
        out = synthetic.detach().clone()
        generator = torch.Generator(device=out.device).manual_seed(202609110000 + int(update))
        for gain, c in ((0.25, "dark_025"), (0.5, "dark_050")):
            mask = gains == gain
            if not mask.any():
                continue
            x = out[mask]
            t = self.tables[c]
            bins = (x * 16).long().clamp(0, 15) * 8 + self.pb

            def lookup(name):
                return torch.stack([t[name][k][bins[:, k]] for k in range(3)], 1)

            z = (
                torch.randn(x.shape, device=x.device, generator=generator)
                if white is None
                else white[mask]
            )
            z = torch.einsum("bchw,dc->bdhw", z, t["source_cholesky"])
            noise = torch.fft.ifft2(torch.fft.fft2(z) * t["filter"].permute(2, 0, 1)).real
            out[mask] = (
                (x + lookup("mean") + lookup("std") * noise).clamp(0, 1) * 255
            ).round() / 255
        return out
