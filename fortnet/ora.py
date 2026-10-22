"""Reliability-weighted transport from fibre cores to the decoder grid."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


class ORA(nn.Module):
    channels = 14

    def __init__(self, geometry, token_width=384, detail_width=8):
        super().__init__()
        with np.load(geometry, allow_pickle=False) as g:
            centers = g["core_centers"].astype(np.float64)
        tree = cKDTree(centers)
        pitch = float(np.median(tree.query(centers, k=2)[0][:, 1]))
        yy, xx = np.mgrid[:92, :92]
        queries = np.stack((1.5 + 4 * yy.ravel(), 1.5 + 4 * xx.ravel()), 1)
        distances, neighbors = tree.query(queries, k=16)
        support = np.exp(-0.5 * np.square(distances / pitch)).astype(np.float32)
        core_dist, core_neighbors = tree.query(centers, k=9)
        self.register_buffer("neighbors", torch.tensor(neighbors, dtype=torch.long))
        self.register_buffer("support", torch.tensor(support, dtype=torch.float32))
        self.register_buffer(
            "core_neighbors", torch.tensor(core_neighbors[:, 1:], dtype=torch.long)
        )
        self.encoder = nn.Sequential(
            nn.Linear(10, token_width),
            nn.GELU(),
            nn.Linear(token_width, token_width),
            nn.GELU(),
            nn.Linear(token_width, token_width),
            nn.GELU(),
            nn.LayerNorm(token_width),
        )
        self.content = nn.Sequential(
            nn.Linear(token_width, token_width // 2),
            nn.GELU(),
            nn.Linear(token_width // 2, 3 + detail_width),
        )
        self.reliability = nn.Sequential(nn.Linear(token_width, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, descriptor):
        with torch.autocast(descriptor.device.type, enabled=False):
            token = self.encoder(descriptor.float())
            content = self.content(token)
            logits = self.reliability(token)
            clip = descriptor[..., -3:].mean(-1)
            norm = descriptor[..., :3]
            neighbor = norm[:, self.core_neighbors]
            variation = (neighbor - norm[:, :, None]).abs().mean((-1, -2))
            consistency = torch.exp(-variation.clamp(0, 8))
            visible = (1.0 - clip).clamp(0.05, 1.0)
            analytic = torch.stack(
                (
                    visible * (0.5 + 0.5 * consistency),
                    visible * (0.25 + 0.75 * consistency),
                    visible * consistency,
                ),
                -1,
            )
            reliability = analytic * (0.25 + 0.75 * logits.sigmoid())
            value = content[:, self.neighbors]
            rel = reliability[:, self.neighbors]
            weight = self.support[None, :, :, None]
            photo_num = (weight * rel * value[..., :3]).sum(-2)
            photo_den = (weight * rel).sum(-2).clamp_min(0.001)
            photo = photo_num / photo_den
            detail_rel = rel[..., 2:3]
            detail_num = (weight * detail_rel * value[..., 3:]).sum(-2)
            detail_den = (weight * detail_rel).sum(-2).clamp_min(0.001)
            detail = detail_num / detail_den
            masks = (weight * rel).sum(-2) / self.support.sum(-1)[None, :, None].clamp_min(0.001)
            grid = torch.cat((photo, detail, masks), -1)
            grid = grid.transpose(1, 2).reshape(len(descriptor), self.channels, 92, 92)
        return torch.nn.functional.interpolate(
            grid, size=(184, 184), mode="bilinear", align_corners=False
        )
