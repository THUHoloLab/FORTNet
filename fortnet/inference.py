"""Reconstruct native MCF frames from a PNG or a directory of PNGs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import read_image
from .evaluation import save_prediction
from .runtime import autocast, load_model, setup


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fortnet_best.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    files = sorted(args.input.glob("*.png")) if args.input.is_dir() else [args.input]
    if not files:
        parser.error("No PNG inputs found.")
    if args.input.is_dir() and args.output.resolve() == args.input.resolve():
        parser.error("Use a separate output directory.")
    device = setup(args.device)
    model, _ = load_model(args.checkpoint, device)
    for start in range(0, len(files), args.batch_size):
        batch = files[start : start + args.batch_size]
        raw = (
            torch.from_numpy(np.stack([read_image(p) for p in batch])).to(device).float().div_(255)
        )
        with autocast(device):
            prediction = model(raw)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite reconstruction.")
        for value, path in zip(prediction, batch):
            destination = args.output / path.name
            if destination.resolve() == path.resolve():
                raise ValueError("Output would overwrite the input image.")
            save_prediction(value, destination)
            print(destination, flush=True)
