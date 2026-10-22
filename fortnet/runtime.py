"""Numerical settings shared by training and inference."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch

from .assets import verify_assets
from .models import FORTNet


def setup(device="cuda", seed=20260909):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for float32 inference.")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The reference mixed-precision protocol requires BF16 support.")
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def initialize(device, seed=20260909):
    verify_assets()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FORTNet().to(device).to(memory_format=torch.channels_last)
    return model


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") not in ("fortnet.weights.v1", "fortnet.training.v1"):
        raise ValueError("Expected a FORTNet release checkpoint.")
    model = initialize(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), payload
