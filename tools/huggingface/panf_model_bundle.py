#!/usr/bin/env python3
"""Build and verify the registered PA-NF Hugging Face model transfer bundle.

The source campaign is external to Git.  This utility selects only the complete
``pathoalign_dep20`` family, verifies it against the promoted artifact index and
the per-cell manifests, includes the required fold standardizers, and fails
atomically on every mismatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCHEMA_VERSION = "panf-model-transfer-bundle/v1"
RELEASE_ID = "panf-model-scorpion-capacity-matched-v1"
REPO_ID = "MatthewVaishnav/paired-acquisition-neural-factorization"
VARIANT = "pathoalign_dep20"
METHOD = "pathoalign"
FOLDS = tuple(range(5))
SEEDS = tuple(range(801, 806))
EXPECTED_COUNT = len(FOLDS) * len(SEEDS)
EXPECTED_EPOCHS = 75
EXPECTED_ATTEMPT = 1
EXPECTED_PARAMETER_COUNT = 1_550_026
TRAINING_SOURCE_COMMIT = "0adea50f1ef22865969109f1834a3c175e3f8b43"
CAMPAIGN_CONFIG_HASH = "59ba04450adf738aa2983dc40ecb4ff09ffa495f13d43c56edde5ec57707a2e1"
SOURCE_MODEL_SHA256 = "528b653459bbdb759c5ca414cf434cf0ad54fba1ef6cbc98e6968c235a3c799f"
SOURCE_MODEL_CRLF_SHA256 = "36855b70b38fc03ea4a798b58c85e08bd6ab7041e30f069b0837cd49fd168d65"

EXPECTED_CONFIG: dict[str, int | float] = {
    "input_dim": 768,
    "biological_dim": 256,
    "acquisition_dim": 64,
    "hidden_dim": 512,
    "temperature": 0.1,
    "reconstruction_weight": 1.0,
    "variance_weight": 1.0,
    "covariance_weight": 0.01,
    "scanner_adversary_weight": 0.5,
    "scanner_acquisition_weight": 0.5,
    "scanner_dependence_weight": 20.0,
    "cross_covariance_weight": 0.05,
    "gradient_reversal_strength": 1.0,
}

EXPECTED_STATE_SHAPES: dict[str, tuple[int, ...]] = {
    "biological.0.weight": (512, 768),
    "biological.0.bias": (512,),
    "biological.2.weight": (512,),
    "biological.2.bias": (512,),
    "biological.3.weight": (256, 512),
    "biological.3.bias": (256,),
    "acquisition.0.weight": (512, 768),
    "acquisition.0.bias": (512,),
    "acquisition.2.weight": (512,),
    "acquisition.2.bias": (512,),
    "acquisition.3.weight": (64, 512),
    "acquisition.3.bias": (64,),
    "scanner_from_b.0.weight": (128, 256),
    "scanner_from_b.0.bias": (128,),
    "scanner_from_b.2.weight": (5, 128),
    "scanner_from_b.2.bias": (5,),
    "scanner_from_a.0.weight": (64, 64),
    "scanner_from_a.0.bias": (64,),
    "scanner_from_a.2.weight": (5, 64),
    "scanner_from_a.2.bias": (5,),
    "decoder.0.weight": (512, 320),
    "decoder.0.bias": (512,),
    "decoder.2.weight": (768, 512),
    "decoder.2.bias": (768,),
}

REQUIRED_INDEX_COLUMNS = {
    "run_id",
    "variant",
    "fold",
    "seed",
    "attempt",
    "status",
    "config_hash",
    "source_commit",
    "cell_manifest_path",
    "cell_manifest_size_bytes",
    "cell_manifest_sha256",
    "checkpoint_path",
    "checkpoint_size_bytes",
    "checkpoint_sha256",
}


class BundleError(RuntimeError):
    """Raised when the PA-NF transfer bundle cannot be proven complete."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, context: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise BundleError(f"Invalid SHA256 for {context}: {value!r}")
    return value


def _repository_root(index_path: Path) -> Path:
    for candidate in (index_path.resolve().parent, *index_path.resolve().parents):
        if (candidate / "LICENSE").is_file() and (
            candidate / "src/models/scorpion_pathoalign.py"
        ).is_file():
            return candidate
    raise BundleError("Could not locate the repository root from the artifact index")


def _load_release_module(root: Path):
    module_path = root / "tools/huggingface/release.py"
    spec = importlib.util.spec_from_file_location("panf_huggingface_release", module_path)
    if spec is None or spec.loader is None:
        raise BundleError(f"Could not load release framework: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise BundleError(
            "PyTorch is required to validate checkpoint contents; install the same "
            "PyTorch major version used by the campaign before building the bundle"
        ) from exc
    return torch


def _safe_results_path(source_results_root: Path, indexed_path: str) -> Path:
    pure = PurePosixPath(indexed_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != "results":
        raise BundleError(f"Unsafe or non-results artifact path: {indexed_path}")
    candidate = source_results_root.joinpath(*pure.parts[1:]).resolve()
    root = source_results_root.resolve()
    if root not in candidate.parents:
        raise BundleError(f"Artifact path escapes the source results root: {indexed_path}")
    return candidate


def _read_index(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            missing = REQUIRED_INDEX_COLUMNS - set(fieldnames)
            if missing:
                raise BundleError(f"Artifact index is missing columns: {sorted(missing)}")
            rows = [dict(row) for row in reader if row["variant"] == VARIANT]
    except OSError as exc:
        raise BundleError(f"Could not read artifact index {path}: {exc}") from exc

    identities: set[tuple[int, int]] = set()
    checkpoint_paths: set[str] = set()
    for row in rows:
        try:
            identity = (int(row["fold"]), int(row["seed"]))
            attempt = int(row["attempt"])
        except ValueError as exc:
            raise BundleError(f"Non-integer fold, seed, or attempt in {row.get('run_id')}") from exc
        if identity in identities:
            raise BundleError(f"Duplicate fold/seed identity in artifact index: {identity}")
        identities.add(identity)
        if row["checkpoint_path"] in checkpoint_paths:
            raise BundleError(f"Duplicate checkpoint path: {row['checkpoint_path']}")
        checkpoint_paths.add(row["checkpoint_path"])
        if row["status"] != "valid":
            raise BundleError(f"Indexed cell is not valid: {row['run_id']}")
        if attempt != EXPECTED_ATTEMPT:
            raise BundleError(f"Unexpected attempt for {row['run_id']}: {attempt}")
        if row["config_hash"] != CAMPAIGN_CONFIG_HASH:
            raise BundleError(f"Campaign config mismatch for {row['run_id']}")
        if row["source_commit"] != TRAINING_SOURCE_COMMIT:
            raise BundleError(f"Training source commit mismatch for {row['run_id']}")
        _require_sha256(row["checkpoint_sha256"], f"checkpoint {row['run_id']}")
        _require_sha256(row["cell_manifest_sha256"], f"cell manifest {row['run_id']}")

    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if len(rows) != EXPECTED_COUNT or identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise BundleError(
            f"Expected the complete {EXPECTED_COUNT}-checkpoint family; "
            f"observed={len(rows)}, missing={missing}, extra={extra}"
        )
    rows.sort(key=lambda row: (int(row["fold"]), int(row["seed"])))
    return fieldnames, rows


def _verify_file(path: Path, expected_size: int, expected_sha256: str, context: str) -> None:
    if not path.is_file():
        raise BundleError(f"Missing {context}: {path}")
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise BundleError(
            f"Size mismatch for {context}: expected={expected_size}, observed={observed_size}, "
            f"path={path}"
        )
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise BundleError(
            f"SHA256 mismatch for {context}: expected={expected_sha256}, "
            f"observed={observed_sha256}, path={path}"
        )


def _load_cell_manifest(path: Path, row: dict[str, str]) -> dict[str, Any]:
    _verify_file(
        path,
        int(row["cell_manifest_size_bytes"]),
        row["cell_manifest_sha256"],
        f"cell manifest {row['run_id']}",
    )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Unreadable cell manifest: {path}") from exc
    expected = {
        "run_id": row["run_id"],
        "variant": VARIANT,
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "attempt": EXPECTED_ATTEMPT,
        "config_hash": CAMPAIGN_CONFIG_HASH,
        "source_commit": TRAINING_SOURCE_COMMIT,
        "campaign_mode": "full",
        "evidence_eligible": True,
        "status": "valid",
        "executed_epochs": EXPECTED_EPOCHS,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise BundleError(
                f"Cell-manifest identity mismatch for {row['run_id']}: "
                f"{name}={manifest.get(name)!r}, expected={value!r}"
            )
    counts = manifest.get("parameter_counts")
    if not isinstance(counts, dict) or counts.get("total") != EXPECTED_PARAMETER_COUNT:
        raise BundleError(f"Parameter-count mismatch in cell manifest {row['run_id']}")
    if counts.get("trainable") != EXPECTED_PARAMETER_COUNT:
        raise BundleError(f"Trainable-parameter mismatch in cell manifest {row['run_id']}")
    standardization_sha256 = manifest.get("standardization_sha256")
    if not isinstance(standardization_sha256, str):
        raise BundleError(f"Missing standardization hash in cell manifest {row['run_id']}")
    _require_sha256(standardization_sha256, f"standardization {row['run_id']}")

    checkpoint_records = [
        record
        for record in manifest.get("artifacts", [])
        if isinstance(record, dict) and str(record.get("path", "")).endswith("checkpoint.pt")
    ]
    if len(checkpoint_records) != 1:
        raise BundleError(f"Expected one checkpoint artifact in cell manifest {row['run_id']}")
    checkpoint_record = checkpoint_records[0]
    if (
        int(checkpoint_record.get("size_bytes", -1)) != int(row["checkpoint_size_bytes"])
        or checkpoint_record.get("sha256") != row["checkpoint_sha256"]
    ):
        raise BundleError(f"Checkpoint index/manifest mismatch for {row['run_id']}")
    return manifest


def _same_config(observed: Any) -> bool:
    if not isinstance(observed, dict) or set(observed) != set(EXPECTED_CONFIG):
        return False
    return all(observed[name] == value for name, value in EXPECTED_CONFIG.items())


def _verify_checkpoint(path: Path, row: dict[str, str], torch_module: Any) -> dict[str, Any]:
    _verify_file(
        path,
        int(row["checkpoint_size_bytes"]),
        row["checkpoint_sha256"],
        f"checkpoint {row['run_id']}",
    )
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            checkpoint = torch_module.load(path, map_location="cpu")
        except Exception as exc:
            raise BundleError(f"torch.load failed for {path}: {exc}") from exc
    except Exception as exc:
        raise BundleError(f"torch.load failed for {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise BundleError(f"Checkpoint is not a mapping: {path}")
    expected_metadata = {
        "method": METHOD,
        "seed": int(row["seed"]),
        "epochs": EXPECTED_EPOCHS,
        "strict_determinism": True,
    }
    for name, value in expected_metadata.items():
        if checkpoint.get(name) != value:
            raise BundleError(
                f"Checkpoint metadata mismatch for {row['run_id']}: "
                f"{name}={checkpoint.get(name)!r}, expected={value!r}"
            )
    if not _same_config(checkpoint.get("config")):
        raise BundleError(f"Checkpoint config mismatch for {row['run_id']}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(EXPECTED_STATE_SHAPES):
        missing = sorted(set(EXPECTED_STATE_SHAPES) - set(state or {}))
        extra = sorted(set(state or {}) - set(EXPECTED_STATE_SHAPES))
        raise BundleError(
            f"State-dict key mismatch for {row['run_id']}: missing={missing}, extra={extra}"
        )
    parameter_count = 0
    for name, expected_shape in EXPECTED_STATE_SHAPES.items():
        tensor = state[name]
        if not torch_module.is_tensor(tensor):
            raise BundleError(f"State-dict value is not a tensor: {row['run_id']}:{name}")
        observed_shape = tuple(int(item) for item in tensor.shape)
        if observed_shape != expected_shape:
            raise BundleError(
                f"State-dict shape mismatch for {row['run_id']}:{name}: "
                f"expected={expected_shape}, observed={observed_shape}"
            )
        if not bool(torch_module.isfinite(tensor).all().item()):
            raise BundleError(f"Non-finite checkpoint tensor: {row['run_id']}:{name}")
        parameter_count += int(tensor.numel())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise BundleError(
            f"State-dict parameter count mismatch for {row['run_id']}: {parameter_count}"
        )
    return checkpoint


def _verify_standardization(path: Path, expected_sha256: str, fold: int) -> dict[str, Any]:
    if not path.is_file():
        raise BundleError(f"Missing fold-{fold} standardization: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise BundleError(
            f"Standardization SHA256 mismatch for fold {fold}: "
            f"expected={expected_sha256}, observed={observed_sha256}, path={path}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"mean", "std"}:
                raise BundleError(f"Unexpected standardization arrays for fold {fold}")
            mean = np.asarray(archive["mean"])
            std = np.asarray(archive["std"])
    except (OSError, ValueError, KeyError) as exc:
        raise BundleError(f"Unreadable standardization for fold {fold}: {path}") from exc
    for name, array in (("mean", mean), ("std", std)):
        if array.shape != (1, EXPECTED_CONFIG["input_dim"]):
            raise BundleError(f"Unexpected {name} shape for fold {fold}: {array.shape}")
        if array.dtype != np.float32:
            raise BundleError(f"Unexpected {name} dtype for fold {fold}: {array.dtype}")
        if not np.isfinite(array).all():
            raise BundleError(f"Non-finite {name} values for fold {fold}")
    if not bool((std > 0).all()):
        raise BundleError(f"Non-positive standardization scale for fold {fold}")
    return {
        "fold": fold,
        "path": f"preprocessing/fold_{fold}_standardization.npz",
        "size_bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "arrays": {"mean": [1, 768], "std": [1, 768]},
        "dtype": "float32",
        "fit_scope": "all non-test slides in registered fold",
    }


def _campaign_prefix(row: dict[str, str]) -> PurePosixPath:
    parts = PurePosixPath(row["checkpoint_path"]).parts
    try:
        cells_index = parts.index("cells")
    except ValueError as exc:
        raise BundleError(
            f"Checkpoint path has no cells directory: {row['checkpoint_path']}"
        ) from exc
    if parts[0] != "results" or cells_index < 2:
        raise BundleError(f"Unexpected checkpoint path layout: {row['checkpoint_path']}")
    return PurePosixPath(*parts[1:cells_index])


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    observed = sha256_file(destination)
    if observed != expected_sha256:
        raise BundleError(
            f"Copied-byte verification failed for {destination}: "
            f"expected={expected_sha256}, observed={observed}"
        )


def _write_filtered_index(
    destination: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(
    index_path: Path, source_results_root: Path, target_bundle: Path
) -> dict[str, Any]:
    index_path = index_path.resolve()
    source_results_root = source_results_root.resolve()
    target_bundle = target_bundle.resolve()
    if target_bundle.exists():
        raise BundleError(f"Refusing to overwrite existing target bundle: {target_bundle}")
    if not source_results_root.is_dir():
        raise BundleError(f"Source results root does not exist: {source_results_root}")

    root = _repository_root(index_path)
    release = _load_release_module(root)
    torch_module = _load_torch()
    fieldnames, rows = _read_index(index_path)
    index_sha256 = sha256_file(index_path)
    model_source = root / "src/models/scorpion_pathoalign.py"
    model_source_sha256 = sha256_file(model_source)
    if model_source_sha256 != SOURCE_MODEL_SHA256:
        raise BundleError(
            f"Exact model-definition hash mismatch: expected={SOURCE_MODEL_SHA256}, "
            f"observed={model_source_sha256}"
        )

    campaign_prefixes = {_campaign_prefix(row) for row in rows}
    if len(campaign_prefixes) != 1:
        raise BundleError(f"Model family spans multiple campaign roots: {campaign_prefixes}")
    campaign_prefix = next(iter(campaign_prefixes))

    target_bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target_bundle.name}.", dir=str(target_bundle.parent))
    )
    checkpoint_records: list[dict[str, Any]] = []
    fold_hashes: dict[int, set[str]] = {fold: set() for fold in FOLDS}
    try:
        card = root / "docs/releases/huggingface/paired-acquisition-neural-factorization/README.md"
        release.validate_card(card, "model")
        tracked_files = {
            card: staging / "README.md",
            root / "LICENSE": staging / "LICENSE",
            model_source: staging / "scorpion_pathoalign.py",
            root / "tools/huggingface/panf_model_inference.py": staging / "inference.py",
        }
        for source, destination in tracked_files.items():
            if not source.is_file():
                raise BundleError(f"Missing tracked release source: {source}")
            _copy_verified(source, destination, sha256_file(source))

        for row in rows:
            fold = int(row["fold"])
            seed = int(row["seed"])
            checkpoint_source = _safe_results_path(source_results_root, row["checkpoint_path"])
            manifest_source = _safe_results_path(source_results_root, row["cell_manifest_path"])
            manifest = _load_cell_manifest(manifest_source, row)
            _verify_checkpoint(checkpoint_source, row, torch_module)
            fold_hashes[fold].add(str(manifest["standardization_sha256"]))

            checkpoint_relative = (
                Path("checkpoints") / f"fold_{fold}" / f"seed_{seed}" / "checkpoint.pt"
            )
            manifest_relative = (
                Path("provenance/source-cell-manifests")
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "cell_manifest.json"
            )
            _copy_verified(
                checkpoint_source,
                staging / checkpoint_relative,
                row["checkpoint_sha256"],
            )
            _copy_verified(
                manifest_source,
                staging / manifest_relative,
                row["cell_manifest_sha256"],
            )
            checkpoint_records.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "run_id": row["run_id"],
                    "variant": VARIANT,
                    "method": METHOD,
                    "attempt": EXPECTED_ATTEMPT,
                    "epochs": EXPECTED_EPOCHS,
                    "strict_determinism": True,
                    "config_hash": CAMPAIGN_CONFIG_HASH,
                    "checkpoint_path": checkpoint_relative.as_posix(),
                    "checkpoint_size_bytes": int(row["checkpoint_size_bytes"]),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "source_cell_manifest_path": manifest_relative.as_posix(),
                    "source_cell_manifest_size_bytes": int(row["cell_manifest_size_bytes"]),
                    "source_cell_manifest_sha256": row["cell_manifest_sha256"],
                    "content_validation": {
                        "torch_load": True,
                        "metadata": True,
                        "config": True,
                        "state_dict_keys_and_shapes": True,
                        "finite_tensors": True,
                        "parameter_count": EXPECTED_PARAMETER_COUNT,
                    },
                }
            )

        preprocessing_records = []
        for fold in FOLDS:
            if len(fold_hashes[fold]) != 1:
                raise BundleError(
                    f"Fold {fold} cells do not identify one standardization hash: "
                    f"{sorted(fold_hashes[fold])}"
                )
            expected_hash = next(iter(fold_hashes[fold]))
            standardization_source = source_results_root.joinpath(
                *campaign_prefix.parts,
                "fold_context",
                f"fold_{fold}_standardization.npz",
            )
            record = _verify_standardization(standardization_source, expected_hash, fold)
            _copy_verified(
                standardization_source,
                staging / record["path"],
                expected_hash,
            )
            preprocessing_records.append(record)

        filtered_index = staging / "provenance/cell_artifact_index.pathoalign_dep20.csv"
        _write_filtered_index(filtered_index, fieldnames, rows)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_id": RELEASE_ID,
            "repo_id": REPO_ID,
            "artifact_type": "model",
            "created_at": utc_now(),
            "package_structure": "complete registered 5-fold x 5-seed final-epoch family",
            "checkpoint_selection": "none; fixed final epoch only",
            "reference_checkpoint": None,
            "checkpoint_count": len(checkpoint_records),
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            "variant": VARIANT,
            "method": METHOD,
            "training_source_commit": TRAINING_SOURCE_COMMIT,
            "campaign_config_hash": CAMPAIGN_CONFIG_HASH,
            "source_artifact_index": {
                "repository_path": index_path.relative_to(root).as_posix(),
                "sha256": index_sha256,
                "filtered_copy": filtered_index.relative_to(staging).as_posix(),
            },
            "model_definition": {
                "path": "scorpion_pathoalign.py",
                "git_blob_sha256": SOURCE_MODEL_SHA256,
                "windows_crlf_sha256": SOURCE_MODEL_CRLF_SHA256,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
            },
            "model_config": EXPECTED_CONFIG,
            "input_contract": {
                "representation": "frozen facebook/dinov2-base CLS feature",
                "backbone_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
                "shape": ["n_samples", 768],
                "dtype": "float32",
                "standardization": "(features - fold_mean) / fold_std",
                "scanner_labels_required_during_training": True,
                "scanner_labels_required_during_inference": False,
                "checkpoint_alone_sufficient": False,
            },
            "preprocessing": preprocessing_records,
            "checkpoints": checkpoint_records,
            "excluded": [
                "raw SCORPION images",
                "frozen DINOv2 feature archive",
                "projected feature arrays",
                "metrics and training-history debris",
                "non-pathoalign_dep20 variants",
            ],
            "evidence_repo": "MatthewVaishnav/paired-acquisition-factorization-evidence",
            "license_gate": {
                "public_release_allowed": False,
                "reason": (
                    "The SCORPION Zenodo v1 record does not state an explicit license. "
                    "Public checkpoint redistribution requires documented permission or "
                    "an explicit compatible dataset license."
                ),
            },
        }
        (staging / "model-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        release.write_checksum_manifest(staging)
        release.verify_checksum_manifest(staging)
        os.replace(staging, target_bundle)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "bundle": str(target_bundle),
        "checkpoints": EXPECTED_COUNT,
        "preprocessing_files": len(FOLDS),
        "verified": True,
        "public_release_allowed": False,
    }


def verify_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise BundleError(f"Bundle directory does not exist: {bundle}")
    root = _repository_root(
        Path(__file__).resolve().parents[2]
        / "evidence/paired_acquisition/scorpion-capacity-matched-20260726/campaign/"
        "cell_artifact_index.csv"
    )
    release = _load_release_module(root)
    release.verify_checksum_manifest(bundle)
    release.validate_card(bundle / "README.md", "model")
    try:
        manifest = json.loads((bundle / "model-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("Bundle model-manifest.json is unreadable") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("Unexpected model bundle schema")
    if manifest.get("release_id") != RELEASE_ID or manifest.get("repo_id") != REPO_ID:
        raise BundleError("Model bundle release identity mismatch")
    checkpoints = manifest.get("checkpoints")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(checkpoints, list) or len(checkpoints) != EXPECTED_COUNT:
        raise BundleError("Model bundle does not contain 25 checkpoint records")
    if not isinstance(preprocessing, list) or len(preprocessing) != len(FOLDS):
        raise BundleError("Model bundle does not contain five preprocessing records")
    torch_module = _load_torch()
    identities: set[tuple[int, int]] = set()
    for record in checkpoints:
        if not isinstance(record, dict):
            raise BundleError("Invalid checkpoint record in model manifest")
        fold, seed = int(record["fold"]), int(record["seed"])
        identities.add((fold, seed))
        row = {
            "run_id": str(record["run_id"]),
            "seed": str(seed),
            "checkpoint_size_bytes": str(record["checkpoint_size_bytes"]),
            "checkpoint_sha256": str(record["checkpoint_sha256"]),
        }
        _verify_checkpoint(bundle / str(record["checkpoint_path"]), row, torch_module)
    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if identities != expected:
        raise BundleError("Checkpoint identity grid is incomplete")
    for record in preprocessing:
        if not isinstance(record, dict):
            raise BundleError("Invalid preprocessing record in model manifest")
        _verify_standardization(
            bundle / str(record["path"]), str(record["sha256"]), int(record["fold"])
        )
    return {
        "bundle": str(bundle),
        "checkpoints": len(checkpoints),
        "preprocessing_files": len(preprocessing),
        "verified": True,
        "public_release_allowed": bool(
            manifest.get("license_gate", {}).get("public_release_allowed", False)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build an atomic verified PA-NF bundle")
    build.add_argument("--artifact-index", type=Path, required=True)
    build.add_argument("--source-results-root", type=Path, required=True)
    build.add_argument("--target-bundle", type=Path, required=True)
    build.set_defaults(
        handler=lambda args: build_bundle(
            args.artifact_index, args.source_results_root, args.target_bundle
        )
    )
    verify = subparsers.add_parser("verify", help="Re-verify a completed PA-NF bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.set_defaults(handler=lambda args: verify_bundle(args.bundle))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (BundleError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
