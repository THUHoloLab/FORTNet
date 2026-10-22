"""Geometry-based observation lifting and multiscale spatial attention."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PixelChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            value = F.layer_norm(
                x.float().permute(0, 2, 3, 1), (self.width,), self.weight, self.bias, 1e-06
            )
        return value.permute(0, 3, 1, 2).to(x.dtype)


def windowize(x, heads, window=8):
    b, c, h, w = x.shape
    ph = -h % window
    pw = -w % window
    x = F.pad(x, (0, pw, 0, ph))
    nh = (h + ph) // window
    nw = (w + pw) // window
    value = x.reshape(b, heads, c // heads, nh, window, nw, window)
    value = value.permute(0, 3, 5, 1, 4, 6, 2).reshape(
        b * nh * nw, heads, window * window, c // heads
    )
    return (value, (b, c, h, w, nh, nw))


def unwindow(value, layout, window=8):
    b, c, h, w, nh, nw = layout
    heads = value.shape[1]
    value = value.reshape(b, nh, nw, heads, window, window, c // heads)
    value = value.permute(0, 3, 6, 1, 4, 2, 5).reshape(b, c, nh * window, nw * window)
    return value[:, :, :h, :w]


class SpatialWindowBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.heads = 4
        self.window = 8
        if width % self.heads:
            raise ValueError("Four complete attention heads required")
        self.norm1 = PixelChannelNorm(width)
        self.norm2 = PixelChannelNorm(width)
        self.qkv = nn.Conv2d(width, 3 * width, 1, bias=False)
        self.local = nn.Conv2d(
            3 * width, 3 * width, 3, padding=1, groups=3 * width, padding_mode="reflect", bias=False
        )
        self.project = nn.Conv2d(width, width, 1)
        self.logit_scale = nn.Parameter(torch.zeros(self.heads, 1, 1))
        self.scale1 = nn.Parameter(torch.full((1, width, 1, 1), 0.1))
        self.expand = nn.Conv2d(width, 2 * width, 1)
        self.reduce = nn.Conv2d(2 * width, width, 1)
        self.scale2 = nn.Parameter(torch.full((1, width, 1, 1), 0.1))
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        pos = torch.stack((yy.flatten(), xx.flatten()), -1)
        delta = pos[:, None] - pos[None, :] + 7
        self.register_buffer("relative_index", delta[:, :, 0] * 15 + delta[:, :, 1])
        self.relative_bias = nn.Parameter(torch.zeros(225, self.heads))

    def attention_inputs(self, x):
        query, key, value = self.local(self.qkv(x)).chunk(3, dim=1)
        q, layout = windowize(query, self.heads)
        k, _ = windowize(key, self.heads)
        v, _ = windowize(value, self.heads)
        q = q * self.logit_scale.exp().to(q.dtype)[None]
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)[None].to(q.dtype)
        b, c, h, w, nh, nw = layout
        if h % 8 or w % 8:
            valid, _ = windowize(torch.ones(1, 1, h, w, device=x.device, dtype=torch.bool), 1)
            valid = valid[:, 0, :, 0]
            bias = (
                bias.expand(nh * nw, -1, -1, -1)
                .masked_fill(~valid[:, None, None, :], float("-inf"))
                .repeat(b, 1, 1, 1)
            )
        return (q, k, v, bias, layout)

    def window_mix(self, x):
        q, k, v, bias, layout = self.attention_inputs(x)
        mixed = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        return self.project(unwindow(mixed, layout))

    def forward(self, x):
        value = self.window_mix(self.norm1(x))
        x = x + value * self.scale1.to(value.dtype)
        value = self.reduce(F.gelu(self.expand(self.norm2(x))))
        return x + value * self.scale2.to(value.dtype)


class SceneBlock(SpatialWindowBlock):
    def __init__(self, width):
        super().__init__(width)
        yy, xx = torch.meshgrid(torch.arange(23), torch.arange(23), indexing="ij")
        pos = torch.stack((yy.flatten(), xx.flatten()), -1)
        delta = pos[:, None] - pos[None, :] + 22
        self.relative_index = delta[:, :, 0] * 45 + delta[:, :, 1]
        self.relative_bias = nn.Parameter(torch.zeros(2025, self.heads))

    def window_mix(self, x):
        b, c, h, w = x.shape
        if (h, w) != (23, 23):
            raise ValueError("Native scene grid must be23x23")
        q, k, v = self.local(self.qkv(x)).chunk(3, 1)

        def tokens(t):
            return t.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)

        q = tokens(q) * self.logit_scale.exp().to(q.dtype)[None]
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)[None].to(q.dtype)
        value = F.scaled_dot_product_attention(
            q, tokens(k), tokens(v), attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        return self.project(value.transpose(-1, -2).reshape(b, c, h, w))


class DetailBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = PixelChannelNorm(width)
        self.expand = nn.Conv2d(width, 2 * width, 1)
        self.spatial = nn.Conv2d(
            2 * width, 2 * width, 5, padding=2, groups=2 * width, padding_mode="reflect"
        )
        self.reduce = nn.Conv2d(2 * width, width, 1)
        self.scale = nn.Parameter(torch.full((1, width, 1, 1), 0.1))

    def forward(self, x):
        value = self.reduce(F.gelu(self.spatial(self.expand(self.norm(x)))))
        return x + value * self.scale.to(value.dtype)


class WindowFibreFieldNet(nn.Module):
    def __init__(self, kind, geometry):
        super().__init__()
        if kind not in ("W",):
            raise ValueError("Unsupported encoder configuration")
        self.kind = kind
        self.node_count = 6406
        with np.load(geometry, allow_pickle=False) as data:
            for name in ("labels", "profile", "support", "reference_mean"):
                self.register_buffer(name, torch.from_numpy(np.array(data[name], copy=True)))
            self.register_buffer(
                "pixel_indices", torch.from_numpy(data["core_pixel_indices"].copy())
            )
            self.register_buffer(
                "pixel_weights", torch.from_numpy(data["core_pixel_weights"].copy())
            )
            coordinates = torch.from_numpy(data["core_pixel_geometry"].copy()).T.reshape(
                1, 3, 368, 368
            )
            self.register_buffer("pixel_geometry", coordinates)
        denominator = torch.zeros(6406, dtype=torch.float32)
        denominator.scatter_add_(0, self.labels.long(), self.profile.square() * self.support)
        self.register_buffer("denominator", denominator.clamp_min(0.0001))
        widths = (96, 160, 256, 512)
        counts = (2, 2, 4, 6)
        self.stem = nn.Conv2d(48, widths[0], 3, padding=1, padding_mode="reflect")
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    *[SceneBlock(w) if i == 3 else SpatialWindowBlock(w) for _ in range(n)]
                )
                for i, (w, n) in enumerate(zip(widths, counts))
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Conv2d(a, b, 3, stride=2, padding=1, padding_mode="reflect")
                for a, b in zip(widths, widths[1:])
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a, 4 * b, 1), nn.PixelShuffle(2))
                for a, b in zip(widths[:0:-1], widths[-2::-1])
            ]
        )
        self.fuse = nn.ModuleList([nn.Conv2d(2 * w, w, 1) for w in widths[-2::-1]])
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(*[SpatialWindowBlock(w) for _ in range(n)])
                for w, n in zip(widths[-2::-1], (4, 2, 2))
            ]
        )
        self.full_synthesis = nn.Sequential(
            nn.Conv2d(widths[0], 4 * 48, 1),
            nn.PixelShuffle(2),
            *[DetailBlock(48) for _ in range(4)],
        )
        self.pixel_output = nn.Conv2d(48, 3, 3, padding=1, padding_mode="reflect")

    def measurements(self, raw):
        if raw.ndim != 4 or raw.shape[1:] != (3, 368, 368):
            raise ValueError("Native Bx3x368x368 RGB required")
        with torch.autocast(raw.device.type, enabled=False):
            raw = raw.float()
            scale = raw.mean((1, 2, 3), keepdim=True).clamp_min(1 / 255)
            normalized = raw * (self.reference_mean / scale)
            index = self.labels.long()[None, None].expand(len(raw), 3, -1)
            numerator = raw.new_zeros((len(raw), 3, 6406))
            numerator.scatter_add_(
                2, index, normalized.flatten(2) * (self.profile * self.support)[None, None]
            )
            nodes = (numerator / self.denominator[None, None]).transpose(1, 2)
            return (normalized, nodes, torch.log(scale / self.reference_mean).flatten(1))

    def lift(self, nodes):
        with torch.autocast(nodes.device.type, enabled=False):
            field = None
            for i in range(4):
                piece = (
                    nodes.float()[:, self.pixel_indices[:, i]]
                    * self.pixel_weights[None, :, i, None]
                )
                field = piece if field is None else field + piece
            return field.transpose(1, 2).reshape(len(nodes), 3, 368, 368)

    def forward(self, raw):
        normalized, nodes, _ = self.measurements(raw)
        field = self.lift(nodes)
        first = raw
        image = torch.cat(
            (first, normalized, field, self.pixel_geometry.expand(len(raw), -1, -1, -1)), dim=1
        )
        packed = F.pixel_unshuffle(image, 2)
        x = self.stem(packed.contiguous(memory_format=torch.channels_last))
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < 3:
                skips.append(x)
                x = self.down[i](x)
        for skip, up, fuse, stage in zip(reversed(skips), self.up, self.fuse, self.decoder):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("Native decoder scale mismatch")
            x = stage(fuse(torch.cat((x, skip), dim=1)))
        return self.pixel_output(self.full_synthesis(x)).float()
