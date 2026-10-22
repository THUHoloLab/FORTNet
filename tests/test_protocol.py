"""Checks for sampling, exposure response, checkpoint state and metrics."""

import hashlib

import numpy as np
import pytest
import torch

from fortnet.assets import verify_assets
from fortnet.checkpoint import pack, restore, save
from fortnet.data import batches
from fortnet.exposure import CalibratedExposure, draws
from fortnet.metrics import metrics
from fortnet.training import joint_step, lr_at


def test_reference_order():
    digest = hashlib.sha256()
    for ids in batches(5200, 65000):
        digest.update(("|".join(f"{i + 1:06d}" for i in ids) + "\n").encode())
    assert digest.hexdigest() == "0e6155e8e50ed7a9ec4c0361194eef1d8ba8c59485f9cea2085b2003cfb05b13"


def test_calibration_and_identity():
    verify_assets()
    sim = CalibratedExposure("cpu")
    raw = torch.linspace(0, 1, 368 * 368).reshape(1, 1, 368, 368).expand(1, 3, -1, -1)
    assert torch.equal(sim.apply(raw, [1], 1), raw)
    for gain in (0.25, 0.5, 0.7, 1.5, 2):
        actual = sim.apply(raw, [gain], 4)
        assert torch.equal(actual, sim.apply(raw, [gain], 4))
        assert torch.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
    with pytest.raises(ValueError):
        sim.apply(raw, [0], 1)
    assert np.array_equal(draws(37, 8), draws(37, 8))


def test_float_metrics():
    torch.manual_seed(5)
    target = torch.rand(1, 3, 16, 16) * 0.5
    result = metrics(target + 0.1, target)
    assert abs(result["psnr"][0] - 20) < 1e-4
    assert abs(metrics(target, target)["ssim"][0] - 1) < 1e-12


def test_schedule_is_not_shortened():
    config = dict(lr=0.0002, lr_min=0.000002, schedule_updates=65000)
    assert lr_at(65000, config) == config["lr_min"]
    assert lr_at(100, config) > 0.000199


def test_resume_matches_continuous(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = torch.nn.Conv2d(3, 3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0)
    raw, target = torch.rand(2, 3, 12, 12), torch.rand(2, 3, 12, 12)
    config = dict(seed=19)
    joint_step(model, optimizer, raw, target, raw * 0.5)
    path = tmp_path / "latest.pt"
    save(path, pack(model, optimizer, config, 1, "sample-order", None, "data"))
    joint_step(model, optimizer, raw, target, raw * 0.5)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    resumed = torch.nn.Conv2d(3, 3, 1)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.0002, weight_decay=0)
    payload = restore(path, resumed, resumed_opt, config, "data")
    assert payload["update"] == 1
    joint_step(resumed, resumed_opt, raw, target, raw * 0.5)
    for key, value in resumed.state_dict().items():
        assert torch.equal(value, expected[key])
    with pytest.raises(ValueError):
        restore(path, resumed, resumed_opt, config, "different-data")


def test_archive_path_validation(tmp_path):
    import zipfile

    from fortnet.download import extract_archive

    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.png", b"")
    with pytest.raises(ValueError):
        extract_archive(path, tmp_path / "data", "nominal/test.zip", 1)


def test_best_commit_recovery(tmp_path):
    from fortnet.checkpoint import ensure_best

    def payload(update, score):
        return dict(
            format="fortnet.training.v1",
            config={"seed": 1},
            data_fingerprint="data",
            update=update,
            best={"update": update, "score": score},
            model={"weight": torch.tensor([float(update)])},
        )

    old, new = payload(1, 20), payload(2, 21)
    save(tmp_path / "best.pt", old)
    save(tmp_path / "latest.pt", new)
    ensure_best(tmp_path, new)
    repaired = torch.load(tmp_path / "best.pt", weights_only=True)
    assert repaired["update"] == 2 and torch.equal(
        repaired["model"]["weight"], new["model"]["weight"]
    )
    (tmp_path / "best.pt").unlink()
    ensure_best(tmp_path, new)
    assert (tmp_path / "best.pt").exists()
    (tmp_path / "best.pt").unlink()
    new["update"] = 3
    with pytest.raises(ValueError):
        ensure_best(tmp_path, new)


def test_resume_directory_identity(tmp_path):
    import json

    from fortnet.checkpoint import validate_resume_directory

    config = {"seed": 1}
    (tmp_path / "config.json").write_text(json.dumps(config))
    validate_resume_directory(tmp_path, tmp_path / "latest.pt", config)
    with pytest.raises(ValueError):
        validate_resume_directory(tmp_path / "other", tmp_path / "latest.pt", config)
    with pytest.raises(ValueError):
        validate_resume_directory(tmp_path, tmp_path / "latest.pt", {"seed": 2})


def test_zero_stop_rejected(monkeypatch, tmp_path):
    import json

    from fortnet.training import main

    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            dict(seed=20260909, batch_size=8, schedule_updates=65000, lr=0.0002, lr_min=0.000002)
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(config),
            "--stop-after-updates",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
