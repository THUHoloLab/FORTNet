"""FORTNet reconstruction network with observation reliability fusion."""

import torch
from torch import nn
from torch.nn import functional as F

from .assets import GEOMETRY
from .gated_local import GatedLocalBlock
from .ora import ORA
from .window_field import WindowFibreFieldNet


class FeatureCompensation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fuse = nn.Conv2d(2 * width, width, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            value = x.float()
            mean = value.mean((2, 3), keepdim=True)
            variance = (value - mean).square().mean((2, 3), keepdim=True)
            normalized = (value - mean) * torch.rsqrt(variance + 1e-05)
        return x + self.fuse(torch.cat((x, normalized.to(x.dtype)), 1))


class FieldDecoder(WindowFibreFieldNet):
    def __init__(self):
        super().__init__("W", GEOMETRY)
        self.encoder[3] = nn.Sequential(*list(self.encoder[3])[:4])
        self.down[0] = nn.Sequential(self.down[0], FeatureCompensation(160))
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a, 4 * b, 1), nn.PixelShuffle(2))
                for a, b in ((512, 384), (384, 256), (256, 160))
            ]
        )
        self.fuse = nn.ModuleList(
            [nn.Conv2d(a, b, 1) for a, b in ((640, 384), (416, 256), (256, 160))]
        )
        self.full_synthesis = nn.Sequential(
            nn.Conv2d(160, 4 * 96, 1), nn.PixelShuffle(2), *[GatedLocalBlock(96) for _ in range(3)]
        )
        self.pixel_output = nn.Conv2d(96, 3, 3, padding=1, padding_mode="reflect")
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(*[GatedLocalBlock(width) for _ in range(count)])
                for width, count in ((384, 4), (256, 2), (160, 2))
            ]
        )


class FORTNet(FieldDecoder):
    def __init__(self):
        super().__init__()
        clip_denominator = torch.zeros_like(self.denominator)
        clip_denominator.scatter_add_(0, self.labels.long(), self.profile * self.support)
        self.register_buffer("clip_denominator", clip_denominator.clamp_min(0.0001))
        self.enabled = True
        if self.enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261003)
                self.ora = ORA(GEOMETRY)
            base = self.fuse[2]
            expanded = nn.Conv2d(base.in_channels + ORA.channels, base.out_channels, 1)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, : base.in_channels].copy_(base.weight)
                expanded.bias.copy_(base.bias)
            self.fuse[2] = expanded
        else:
            self.ora = None

    def _clip_nodes(self, raw):
        with torch.autocast(raw.device.type, enabled=False):
            values = raw.new_zeros((len(raw), 3, self.node_count))
            index = self.labels.long()[None, None].expand(len(raw), 3, -1)
            values.scatter_add_(
                2,
                index,
                (raw >= 254.5 / 255).float().flatten(2) * (self.profile * self.support)[None, None],
            )
            return (values / self.clip_denominator[None, None]).transpose(1, 2)

    def _descriptor(self, nodes, raw, logscale):
        with torch.autocast(raw.device.type, enabled=False):
            raw_nodes = nodes.float() * logscale.float().exp()[:, None]
            basis = raw.new_tensor(
                (
                    (0.2126, 0.7152, 0.0722),
                    (0.95854625, -0.28493699, 0.0),
                    (0.02744402, 0.09232345, -0.9953508),
                )
            )
            norm_coord = torch.matmul(nodes.float(), basis.t())
            raw_coord = torch.matmul(raw_nodes, basis.t())
            clip = self._clip_nodes(raw)
            scale = logscale.float()[:, None].expand(-1, nodes.shape[1], -1)
            return torch.cat((torch.asinh(norm_coord), torch.asinh(raw_coord), scale, clip), -1)

    def encode(self, raw):
        normalized, nodes, logscale = self.measurements(raw)
        field = self.lift(nodes)
        image = torch.cat(
            (raw, normalized, field, self.pixel_geometry.expand(len(raw), -1, -1, -1)), 1
        )
        x = self.stem(F.pixel_unshuffle(image, 2).contiguous(memory_format=torch.channels_last))
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < 3:
                skips.append(x)
                x = self.down[i](x)
        descriptor = self._descriptor(nodes, raw, logscale) if self.enabled else None
        for i, (skip, up, fuse, stage) in enumerate(
            zip(reversed(skips), self.up, self.fuse, self.decoder)
        ):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("Native decoder scale mismatch")
            if i == 2 and self.enabled:
                obs = self.ora(descriptor)
                x = stage(fuse(torch.cat((x, skip, obs.to(x.dtype)), 1)))
            else:
                x = stage(fuse(torch.cat((x, skip), 1)))
        return x

    def readout(self, x):
        return self.pixel_output(self.full_synthesis(x)).float()

    def forward(self, raw):
        return self.readout(self.encode(raw))
