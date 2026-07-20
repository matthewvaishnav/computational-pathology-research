from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.paired_acquisition_provenance import (
    ProvenanceValidationError,
    validate_release,
)
from src.paired_acquisition_release_writer import write_single_run_release


def build_release(tmp_path: Path) -> Path:
    dataset_source = tmp_path / "source_features.npz"
    split_manifest = tmp_path / "fold_0_patch_manifest.csv"
    projected_features = tmp_path / "projected_features.npz"
    checkpoint = tmp_path / "checkpoint.pt"
    pair_assignments = tmp_path / "pair_assignments.csv"
    training_history = tmp_path / "training_history.csv"
    dataset_source.write_bytes(b"source-feature-archive\n")
    split_manifest.write_text("sample_id,split\ns1,train\ns2,test\n", encoding="utf-8")
    projected_features.write_bytes(b"projected-feature-archive\n")
    checkpoint.write_bytes(b"model-checkpoint\n")
    pair_assignments.write_text("sample_id,pair_id\ns1,p1\ns2,p1\n", encoding="utf-8")
    training_history.write_text("epoch,loss\n1,0.5\n", encoding="utf-8")

    release_dir = tmp_path / "release"
    summary = write_single_run_release(
        output_dir=release_dir,
        code_commit="1" * 40,
        producer_command=["python", "real_producer.py", "--fold", "0"],
        seed=911,
        dataset_name="real-paired-acquisition-fixture",
        dataset_source=dataset_source,
        split_manifest=split_manifest,
        config_payload={
            "acquisition_dim": 8,
            "cross_covariance_weight": 0.05,
            "epochs": 1,
        },
        environment_payload={"python": "3.10", "torch": "fixture"},
        features=projected_features,
        checkpoint=checkpoint,
        additional_artifacts=[
            {
                "role": "pair_assignments",
                "kind": "metadata",
                "source": pair_assignments,
                "path": "pair_assignments.csv",
            },
            {
                "role": "training_history",
                "kind": "output",
                "source": training_history,
                "path": "training_history.csv",
            },
        ],
        metrics_payload={"biological_scanner_accuracy": 0.2},
        run_log_payload={"events": [{"event": "complete"}]},
        feature_metadata={
            "artifact_path": "attempted-override.npz",
            "artifact_sha256": "0" * 64,
            "checkpoint_path": "attempted-override.pt",
            "biological_shape": [2, 8],
            "acquisition_shape": [2, 8],
        },
        created_at="2026-07-20T18:00:00Z",
    )
    assert summary["status"] == "valid"
    assert summary["run_count"] == 1
    return release_dir


def test_real_producer_release_is_self_contained_and_checkpoint_bound(
    tmp_path: Path,
) -> None:
    release_dir = build_release(tmp_path)
    summary = validate_release(release_dir / "release_manifest.json")
    run_id = summary["run_ids"][0]
    run_dir = release_dir / "runs" / run_id

    feature_metadata = json.loads(
        (run_dir / "feature_metadata.json").read_text(encoding="utf-8")
    )["payload"]
    assert feature_metadata["artifact_path"] == "features.npz"
    assert feature_metadata["checkpoint_path"] == "checkpoint.pt"
    assert feature_metadata["artifact_sha256"] != "0" * 64

    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    roles = {artifact["role"] for artifact in record["artifacts"]}
    assert {
        "checkpoint",
        "pair_assignments",
        "training_history",
        "config",
        "dataset_manifest",
        "dataset_source",
        "environment",
        "feature_metadata",
        "features",
        "metrics",
        "run_log",
        "split_manifest",
    }.issubset(roles)
    assert (run_dir / "pair_assignments.csv").is_file()
    assert (run_dir / "training_history.csv").is_file()


def test_real_producer_release_fails_after_artifact_corruption(tmp_path: Path) -> None:
    release_dir = build_release(tmp_path)
    manifest = json.loads(
        (release_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    run_id = manifest["runs"][0]["run_id"]
    metrics_path = release_dir / "runs" / run_id / "metrics.json"
    metrics_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProvenanceValidationError, match="artifact checksum mismatch"):
        validate_release(release_dir / "release_manifest.json")


def test_real_producer_release_refuses_overwrite(tmp_path: Path) -> None:
    release_dir = build_release(tmp_path)
    with pytest.raises(ProvenanceValidationError, match="refusing to overwrite"):
        write_single_run_release(
            output_dir=release_dir,
            code_commit="1" * 40,
            producer_command=["python", "producer.py"],
            seed=1,
            dataset_name="duplicate",
            dataset_source=tmp_path / "source_features.npz",
            split_manifest=tmp_path / "fold_0_patch_manifest.csv",
            config_payload={},
            environment_payload={},
            features=tmp_path / "projected_features.npz",
            checkpoint=tmp_path / "checkpoint.pt",
            metrics_payload={},
            run_log_payload={},
        )
