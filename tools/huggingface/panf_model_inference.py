"""Checksum-aware inference helpers for the PA-NF SCORPION model family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scorpion_pathoalign import ProjectionConfig, ScorpionProjection


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(root: Path, relative_path: str, expected_sha256: str) -> Path:
    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    if resolved_root not in candidate.parents or not candidate.is_file():
        raise ValueError(f"Missing or unsafe release file: {relative_path}")
    observed = _sha256_file(candidate)
    if observed != expected_sha256:
        raise ValueError(
            f"Release-file SHA256 mismatch for {relative_path}: "
            f"expected={expected_sha256}, observed={observed}"
        )
    return candidate


def load_panf(
    release_root: str | Path,
    *,
    fold: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> tuple[ScorpionProjection, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one prespecified family member and its required fold standardizer.

    The 25 checkpoints are co-equal registered final-epoch instances.  ``fold``
    and ``seed`` must be chosen without ranking them on the published test results.
    """

    root = Path(release_root).resolve()
    manifest_path = root / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in manifest["checkpoints"]
        if int(record["fold"]) == int(fold) and int(record["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint for fold={fold}, seed={seed}")
    record = matches[0]
    preprocessing = [item for item in manifest["preprocessing"] if int(item["fold"]) == int(fold)]
    if len(preprocessing) != 1:
        raise ValueError(f"Expected one standardizer for fold={fold}")
    preprocessing_record = preprocessing[0]

    checkpoint_path = _verified_path(root, record["checkpoint_path"], record["checkpoint_sha256"])
    standardization_path = _verified_path(
        root, preprocessing_record["path"], preprocessing_record["sha256"]
    )
    with np.load(standardization_path, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float32)
        std = np.asarray(archive["std"], dtype=np.float32)
    if mean.shape != (1, 768) or std.shape != (1, 768) or not bool((std > 0).all()):
        raise ValueError("Invalid fold standardization arrays")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "pathoalign" or int(checkpoint.get("seed")) != int(seed):
        raise ValueError("Checkpoint method/seed identity mismatch")
    if int(checkpoint.get("epochs")) != 75 or checkpoint.get("strict_determinism") is not True:
        raise ValueError("Checkpoint schedule/determinism identity mismatch")
    config = ProjectionConfig(**checkpoint["config"])
    if config.input_dim != 768 or config.biological_dim != 256:
        raise ValueError("Checkpoint input/biological dimension mismatch")
    if config.acquisition_dim != 64 or config.hidden_dim != 512:
        raise ValueError("Checkpoint acquisition/hidden dimension mismatch")
    model = ScorpionProjection("pathoalign", config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, mean, std, record


def project_features(
    model: ScorpionProjection,
    raw_frozen_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Project raw 768-D frozen features through both PA-NF branches."""

    features = np.asarray(raw_frozen_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 768:
        raise ValueError(f"Expected an [n, 768] feature matrix, observed {features.shape}")
    if len(features) == 0:
        raise ValueError("At least one frozen feature row is required")
    standardized = ((features - mean) / std).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardization produced non-finite values")
    outputs: dict[str, list[np.ndarray]] = {
        "biological": [],
        "acquisition": [],
        "reconstruction": [],
    }
    with torch.inference_mode():
        for start in range(0, len(standardized), batch_size):
            batch = torch.from_numpy(standardized[start : start + batch_size]).to(device)
            projected = model(batch)
            for name in outputs:
                outputs[name].append(projected[name].detach().cpu().numpy())
    return {name: np.concatenate(values).astype(np.float32) for name, values in outputs.items()}
