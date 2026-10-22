"""Train FORTNet with paired nominal and calibrated synthetic views."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .assets import verify_assets
from .checkpoint import ensure_best, pack, restore, save, validate_resume_directory
from .data import ResidentPairs, batches
from .evaluation import evaluate
from .exposure import CalibratedExposure, draws
from .runtime import autocast, initialize, setup


def lr_at(update, config):
    return config["lr_min"] + (config["lr"] - config["lr_min"]) * 0.5 * (
        1 + math.cos(math.pi * update / config["schedule_updates"])
    )


def joint_step(model, optimizer, raw, target, augmented):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = []
    for weight, view in ((0.75, raw), (0.25, augmented)):
        with autocast(raw.device):
            prediction = model(view)
        loss = (prediction.float() - target).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reconstruction loss.")
        (weight * loss).backward()
        values.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return dict(
        loss=0.75 * values[0] + 0.25 * values[1], nominal_loss=values[0], synthetic_loss=values[1]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="Pause at this absolute update without changing the LR horizon.",
    )
    parser.add_argument(
        "--limit-train", type=int, help="Use a small subset for a smoke test; not a reference run."
    )
    parser.add_argument("--save-every", type=int, default=650)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["seed"] != 20260909
        or config["batch_size"] < 1
        or config["schedule_updates"] != 65000
    ):
        parser.error("Use seed 20260909, a positive batch size and the fixed 65000-update horizon.")
    if not 0 <= config["lr_min"] <= config["lr"] or args.save_every < 1:
        parser.error("Invalid learning rate or save interval.")
    config["assets"] = verify_assets()
    config["limit_train"] = args.limit_train
    config["implementation"] = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
    ).hexdigest()
    target_update = (
        config["schedule_updates"] if args.stop_after_updates is None else args.stop_after_updates
    )
    if not 1 <= target_update <= config["schedule_updates"]:
        parser.error("Stop update must be between 1 and 65000.")
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error("Output is not empty. Use --resume or a new output directory.")
    if args.resume:
        validate_resume_directory(args.output, args.resume, config)
    device = setup(args.device, config["seed"])
    if device.type != "cuda":
        parser.error("Training requires a CUDA device with BF16 support.")
    print("Loading and verifying nominal training pairs...", flush=True)
    cache = ResidentPairs(
        args.data, device if args.cache_device == "cuda" else "cpu", args.limit_train
    )
    if len(cache.ids) % config["batch_size"]:
        parser.error("Training sample count must be divisible by batch size.")
    epoch_steps = len(cache.ids) // config["batch_size"]
    model = initialize(device, config["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    simulator = CalibratedExposure(device)
    start = 0
    best = None
    previous_hash = None
    if args.resume:
        payload = restore(args.resume, model, optimizer, config, cache.fingerprint)
        ensure_best(args.output, payload)
        start, best, previous_hash = payload["update"], payload["best"], payload["order_hash"]
        del payload
    if start >= target_update:
        parser.error("The checkpoint has already reached the requested update.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"Train | resume={start} | target={target_update} | batch={config['batch_size']} | samples={len(cache.ids)}",
        flush=True,
    )
    if args.limit_train is not None or config["batch_size"] != 8:
        print(
            "Non-reference run: training subset or batch size differs from the published protocol.",
            flush=True,
        )
    order = hashlib.sha256()
    began = time.monotonic()
    with (args.output / "train.log").open("a", encoding="utf-8", buffering=1) as log:

        def report(message):
            print(message, flush=True)
            log.write(message + "\n")

        for update, ids in enumerate(
            batches(len(cache.ids), target_update, config["batch_size"], config["seed"]), 1
        ):
            order.update(("|".join(cache.ids[i] for i in ids) + "\n").encode())
            if update <= start:
                if update == start and order.hexdigest() != previous_hash:
                    raise ValueError("Resumed sample order differs from the checkpoint.")
                continue
            raw, target = cache.batch(ids, device)
            augmented = simulator.apply(raw, draws(update, len(raw)), update)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(update, config)
            losses = joint_step(model, optimizer, raw, target, augmented)
            if update == start + 1 or update % 25 == 0:
                report(
                    f"Epoch {(update - 1) // epoch_steps + 1:03d} | step {(update - 1) % epoch_steps + 1}/{epoch_steps} | update {update} | loss {losses['loss']:.6f} | lr {optimizer.param_groups[0]['lr']:.8f}"
                )
            improved = False
            if update % epoch_steps == 0:
                result = evaluate(model, args.data, "nominal", "validation", device, 8)
                score = result["summary"]["psnr"]["mean"]
                improved = best is None or score > best["score"]
                if improved:
                    best = dict(score=score, update=update, epoch=update // epoch_steps)
                report(
                    f"Epoch {update // epoch_steps:03d} completed | validation PSNR {score:.6f} | SSIM {result['summary']['ssim']['mean']:.6f} | best {best['score']:.6f} epoch {best['epoch']:03d}"
                )
                (args.output / f"validation_{update // epoch_steps:03d}.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
            if (
                improved
                or update % epoch_steps == 0
                or update % args.save_every == 0
                or update == target_update
            ):
                payload = pack(
                    model, optimizer, config, update, order.hexdigest(), best, cache.fingerprint
                )
                save(args.output / "latest.pt", payload)
                if improved:
                    save(args.output / "best.pt", payload)
                del payload
        if target_update == 65000 and len(cache.ids) == 5200 and config["batch_size"] == 8:
            if (
                order.hexdigest()
                != "0e6155e8e50ed7a9ec4c0361194eef1d8ba8c59485f9cea2085b2003cfb05b13"
            ):
                raise ValueError("Reference sample-order checksum mismatch.")
        report(
            f"Saved latest.pt | update {target_update} | elapsed {time.monotonic() - began:.1f}s"
        )
