from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ['UAV_DATA_ROOT'])
DEFAULT_DATA_ROOT = PROJECT_ROOT / "rtdetrv2_pilot_dataset"
UPSTREAM_ROOT = Path(__file__).resolve().parents[2] / 'vendor/rtdetr'
UPSTREAM_PYTHON_ROOT = UPSTREAM_ROOT
DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "checkpoints"
    / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)

MODEL_COMMIT = "a21d516aca15da57e65f35c47659c7535ad2b6b3"
MODEL_SHORT_COMMIT = "a21d516"
MODEL_REPOSITORY = "https://github.com/lyuwenyu/RT-DETR"
CHECKPOINT_FILENAME = "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
CHECKPOINT_URL = (
    "https://github.com/lyuwenyu/storage/releases/download/v0.2/"
    + CHECKPOINT_FILENAME
)
CHECKPOINT_SHA256 = "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"

COCO_STATS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def add_upstream_to_path() -> None:
    if not UPSTREAM_PYTHON_ROOT.is_dir():
        raise RuntimeError(f"Missing upstream RT-DETR checkout: {UPSTREAM_PYTHON_ROOT}")
    path = str(UPSTREAM_PYTHON_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composite_hash(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(hashes.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_upstream_commit():
    return MODEL_COMMIT


def verify_checkpoint(path: Path) -> str:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing official pretrained checkpoint: {path}")
    actual = sha256_file(path)
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            "Official checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, found {actual} at {path}"
        )
    return actual


def config_sha256(path: Path) -> str:
    return sha256_file(path.resolve())


def load_model_weights(model, checkpoint_path: Path) -> dict[str, Any]:
    """Load official tuning weights and report every compatibility decision."""
    import torch

    verify_checkpoint(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        source_name = "ema.module"
        params = state["ema"]["module"]
    elif "model" in state:
        source_name = "model"
        params = state["model"]
    else:
        raise RuntimeError("Checkpoint contains neither ema.module nor model state")

    destination = model.state_dict()
    matched = {}
    missing = []
    shape_mismatch = []
    for key, value in destination.items():
        if key not in params:
            missing.append(key)
        elif tuple(value.shape) != tuple(params[key].shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "model_shape": list(value.shape),
                    "checkpoint_shape": list(params[key].shape),
                }
            )
        else:
            matched[key] = params[key]
    unexpected = sorted(set(params) - set(destination))
    incompatible = model.load_state_dict(matched, strict=False)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_state": source_name,
        "matched_key_count": len(matched),
        "model_key_count": len(destination),
        "checkpoint_key_count": len(params),
        "missing_keys": sorted(missing),
        "shape_mismatches": sorted(shape_mismatch, key=lambda item: item["key"]),
        "unexpected_checkpoint_keys": unexpected,
        "load_state_missing_keys": sorted(incompatible.missing_keys),
        "load_state_unexpected_keys": sorted(incompatible.unexpected_keys),
    }


def coco_metrics(values: list[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(COCO_STATS, values)}
