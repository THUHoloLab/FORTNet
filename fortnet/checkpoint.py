"""Atomic training checkpoints with optimizer and random state."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def pack(model, optimizer, config, update, order_hash, best, data_fingerprint):
    state = np.random.get_state()
    return dict(
        format="fortnet.training.v1",
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        config=config,
        update=update,
        order_hash=order_hash,
        best=best,
        data_fingerprint=data_fingerprint,
        rng=dict(
            cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            python=random.getstate(),
            numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
        ),
    )


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_resume_directory(output, checkpoint, config):
    output = Path(output).resolve()
    if Path(checkpoint).resolve() != output / "latest.pt":
        raise ValueError(
            "Resume from latest.pt in the same output directory; move the entire run directory together if needed."
        )
    import json

    saved_config = output / "config.json"
    if not saved_config.is_file() or json.loads(saved_config.read_text()) != config:
        raise ValueError("Output directory does not belong to this training configuration.")


def ensure_best(output, latest):
    """Complete a best-file write interrupted after the latest-file commit."""
    path = Path(output) / "best.pt"
    selected = latest["best"]
    if selected is None:
        if path.exists():
            raise ValueError("Unexpected best checkpoint in an unselected run.")
        return
    existing = None
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if (
            existing.get("format") != "fortnet.training.v1"
            or existing["config"] != latest["config"]
            or existing["data_fingerprint"] != latest["data_fingerprint"]
        ):
            raise ValueError("Best checkpoint belongs to a different run.")
        if existing["best"] == selected and existing["update"] == selected["update"]:
            return
        if existing["update"] >= selected["update"]:
            raise ValueError("Best and latest checkpoint selections are inconsistent.")
    if latest["update"] == selected["update"]:
        save(path, latest)
    else:
        raise ValueError(
            "Selected best weights are missing or stale. Restore the matching best.pt before resuming."
        )


def restore(path, model, optimizer, config, data_fingerprint):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "fortnet.training.v1":
        raise ValueError("Resume requires a full training checkpoint, not inference weights.")
    if payload["config"] != config or payload["data_fingerprint"] != data_fingerprint:
        raise ValueError("Resume configuration or training data differs from the saved run.")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    update = payload["update"]
    if {int(v["step"]) for v in optimizer.state.values()} != {update}:
        raise ValueError("Optimizer update count is inconsistent.")
    torch.set_rng_state(payload["rng"]["cpu"])
    if payload["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    random.setstate(payload["rng"]["python"])
    state = payload["rng"]["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    return payload
