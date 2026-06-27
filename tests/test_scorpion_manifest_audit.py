from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scorpion" / "audit_scorpion_manifest.py"
SPEC = importlib.util.spec_from_file_location("audit_scorpion_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SCANNERS = MODULE.EXPECTED_SCANNERS
AuditError = MODULE.AuditError


def make_manifest(tmp_path: Path, *, leak_slide: bool = False) -> Path:
    rows = []
    for slide_index in range(2):
        slide_id = f"slide_{slide_index:02d}"
        for region_index in range(2):
            region_id = f"{slide_id}_region_{region_index:02d}"
            for scanner_id in SCANNERS:
                relative = Path("images") / scanner_id / f"{region_id}.png"
                absolute = tmp_path / relative
                absolute.parent.mkdir(parents=True, exist_ok=True)
                absolute.write_bytes(f"{slide_id}|{region_id}|{scanner_id}".encode("utf-8"))
                split = "train" if slide_index == 0 else "test"
                if leak_slide and slide_index == 0 and region_index == 1:
                    split = "test"
                rows.append(
                    {
                        "slide_id": slide_id,
                        "region_id": region_id,
                        "scanner_id": scanner_id,
                        "path": str(relative),
                        "split": split,
                    }
                )

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_complete_groups_and_slide_grouped_splits_pass(tmp_path: Path):
    manifest = make_manifest(tmp_path)
    out_dir = tmp_path / "audit"

    MODULE.run_audit(
        manifest=manifest,
        data_root=tmp_path,
        out_dir=out_dir,
        strict_release_counts=False,
        check_images=False,
        checksums=True,
    )

    validated = pd.read_csv(out_dir / "validated_manifest.csv")
    pairs = pd.read_csv(out_dir / "scanner_pair_coverage.csv")
    splits = pd.read_csv(out_dir / "split_summary.csv")

    assert len(validated) == 2 * 2 * 5
    assert len(pairs) == 10
    assert (pairs["n_aligned_regions"] == 4).all()
    assert set(splits["split"]) == {"train", "test"}
    assert (out_dir / "scorpion_manifest_audit.md").is_file()


def test_original_slide_leakage_is_rejected(tmp_path: Path):
    manifest = make_manifest(tmp_path, leak_slide=True)
    frame = MODULE.resolve_paths(MODULE.load_manifest(manifest), tmp_path)
    frame["scanner_id"] = frame["scanner_id"].map(MODULE.normalize_scanner)

    with pytest.raises(AuditError, match="Original-slide leakage"):
        MODULE.validate_split_leakage(frame)


def test_incomplete_five_scanner_region_is_rejected(tmp_path: Path):
    manifest = make_manifest(tmp_path)
    frame = MODULE.resolve_paths(MODULE.load_manifest(manifest), tmp_path)
    frame["scanner_id"] = frame["scanner_id"].map(MODULE.normalize_scanner)
    frame = frame.iloc[:-1].copy()

    with pytest.raises(AuditError, match="Incomplete or malformed"):
        MODULE.validate_region_groups(frame)


def test_strict_release_counts_reject_toy_manifest(tmp_path: Path):
    manifest = make_manifest(tmp_path)
    frame = MODULE.resolve_paths(MODULE.load_manifest(manifest), tmp_path)
    frame["scanner_id"] = frame["scanner_id"].map(MODULE.normalize_scanner)
    region_summary = MODULE.validate_region_groups(frame)

    with pytest.raises(AuditError, match="Release-count validation failed"):
        MODULE.validate_slide_structure(
            frame,
            region_summary,
            strict_release_counts=True,
        )
