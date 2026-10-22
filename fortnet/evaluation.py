"""Evaluate one fixed checkpoint on paired real exposure images."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import CONDITIONS, checksums, pairs, read_image
from .metrics import metrics, summarize
from .runtime import autocast, load_model, setup


def save_prediction(prediction, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        prediction.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


@torch.inference_mode()
def evaluate(model, root, condition, split, device, batch_size=8, output=None, limit=None):
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    entries = pairs(root, condition, split)
    if limit is not None:
        if not 1 <= limit <= len(entries):
            raise ValueError("Invalid evaluation subset size.")
        entries = entries[:limit]
    expected = checksums()
    rows = []
    model.eval()
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        raw = (
            torch.from_numpy(
                np.stack(
                    [read_image(a, expected[f"{condition}/{split}/{a.name}"]) for a, _ in chunk]
                )
            )
            .to(device)
            .float()
            .div_(255)
        )
        target = (
            torch.from_numpy(
                np.stack([read_image(b, expected[f"gt/{split}/{b.name}"]) for _, b in chunk])
            )
            .to(device)
            .float()
            .div_(255)
        )
        with autocast(device):
            prediction = model(raw)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite reconstruction.")
        values = metrics(prediction, target)
        for i, (a, _) in enumerate(chunk):
            rows.append(dict(identity=a.stem, **{k: v[i] for k, v in values.items()}))
            if output is not None:
                save_prediction(prediction[i], Path(output) / condition / a.name)
    return dict(count=len(rows), summary=summarize(rows), per_image=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fortnet_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke-test subset; not a full benchmark.")
    args = parser.parse_args()
    device = setup(args.device)
    model, payload = load_model(args.checkpoint, device)
    report = dict(
        split=args.split,
        update=payload["update"],
        batch_size=args.batch_size,
        precision="bfloat16" if device.type == "cuda" else "float32",
        subset=args.limit is not None,
        conditions={},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        result = evaluate(
            model,
            args.data,
            condition,
            args.split,
            device,
            args.batch_size,
            args.output / "predictions" if args.save_images else None,
            args.limit,
        )
        report["conditions"][condition] = result
        print(
            f"{condition:10s} | n={result['count']} | PSNR {result['summary']['psnr']['mean']:.6f} | SSIM {result['summary']['ssim']['mean']:.6f}",
            flush=True,
        )
        (args.output / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
