"""Packaged calibration resources and integrity checks."""

import hashlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent / "assets"
GEOMETRY = DIRECTORY / "geometry.npz"
RESIDUAL = DIRECTORY / "dark_residual.npz"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets():
    expected = json.loads((DIRECTORY / "manifest.json").read_text())
    for name, digest in expected.items():
        if sha256(DIRECTORY / name) != digest:
            raise ValueError(f"Calibration resource checksum mismatch: {name}")
    return expected


def load_response():
    verify_assets()
    return json.loads((DIRECTORY / "response.json").read_text())
