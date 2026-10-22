"""Paired native-resolution PNG loading and deterministic minibatches."""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .assets import DIRECTORY

CONDITIONS = ("nominal", "dark_025", "dark_050", "bright_150", "bright_200")
SPLITS = {"train": 5200, "validation": 400, "test": 400}


def checksums():
    return dict(
        (line.split("  ", 1)[1], line.split("  ", 1)[0])
        for line in (DIRECTORY / "data_checksums.sha256").read_text().splitlines()
    )


def read_image(path, expected=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    with Image.open(io.BytesIO(content)) as image:
        if image.mode != "RGB" or image.size != (368, 368):
            raise ValueError(f"Expected a native 368 x 368 RGB PNG: {path}")
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()


def pairs(root, condition, split):
    if condition not in CONDITIONS or split not in SPLITS:
        raise ValueError("Unknown dataset condition or split.")
    if split == "train" and condition != "nominal":
        raise ValueError("Neural training accepts only nominal inputs.")
    root = Path(root)
    inputs = sorted((root / condition / split).glob("*.png"))
    targets = sorted((root / "gt" / split).glob("*.png"))
    if len(inputs) != SPLITS[split] or [p.name for p in inputs] != [p.name for p in targets]:
        raise ValueError(
            f"Expected {SPLITS[split]} exactly paired PNGs in {condition}/{split} and gt/{split}."
        )
    return list(zip(inputs, targets))


class ResidentPairs:
    def __init__(self, root, device="cpu", limit=None):
        entries = pairs(root, "nominal", "train")
        if limit is not None:
            if not 1 <= limit <= len(entries):
                raise ValueError("Invalid training subset size.")
            entries = entries[:limit]
        expected = checksums()
        self.ids = tuple(a.stem for a, _ in entries)
        self.raw = torch.empty((len(entries), 3, 368, 368), dtype=torch.uint8, device=device)
        self.target = torch.empty_like(self.raw)
        digest = hashlib.sha256()

        def decode(pair):
            a, b = pair
            return (
                read_image(a, expected[f"nominal/train/{a.name}"]),
                read_image(b, expected[f"gt/train/{b.name}"]),
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(entries), 64):
                decoded = list(pool.map(decode, entries[start : start + 64]))
                for offset, (a, b) in enumerate(decoded):
                    digest.update(a.tobytes())
                    digest.update(b.tobytes())
                self.raw[start : start + len(decoded)].copy_(
                    torch.from_numpy(np.stack([x[0] for x in decoded]))
                )
                self.target[start : start + len(decoded)].copy_(
                    torch.from_numpy(np.stack([x[1] for x in decoded]))
                )
        self.fingerprint = digest.hexdigest()

    def batch(self, indices, device):
        index = torch.as_tensor(indices, dtype=torch.long, device=self.raw.device)
        return (
            self.raw.index_select(0, index).to(device).float().div_(255),
            self.target.index_select(0, index).to(device).float().div_(255),
        )


def batches(count, updates, batch_size=8, seed=20260909):
    if count < batch_size or count % batch_size:
        raise ValueError("Training set size must be a positive multiple of batch size.")
    generator = torch.Generator().manual_seed(seed)
    emitted = 0
    while emitted < updates:
        order = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count, batch_size):
            if emitted == updates:
                return
            yield order[start : start + batch_size]
            emitted += 1
