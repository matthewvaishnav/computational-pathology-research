"""
Comprehensive tests for benchmark_manifest.py

Tests cover:
- BenchmarkEntry dataclass creation
- BenchmarkManifest initialization
- Adding entries (add_entry, update_or_add_entry)
- Reading entries (read_all, find_by_experiment)
- Markdown export (to_markdown)
- Error handling (corrupted JSON, invalid schema)
"""

import json
import os
from pathlib import Path

import pytest

from src.utils.benchmark_manifest import BenchmarkEntry, BenchmarkManifest


# ============================================================================
# BENCHMARK ENTRY TESTS
# ============================================================================


def test_benchmark_entry_creation():
    """Test BenchmarkEntry dataclass creation."""
    entry = BenchmarkEntry(
        experiment_name="test_exp",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="configs/test.yaml",
        train_command="python train.py",
        eval_command="python eval.py",
        final_metrics={"accuracy": 0.95, "auc": 0.98},
        artifact_paths={"model": "checkpoints/model.pth"},
        caveats=["Small dataset", "No validation"],
        notes="Test experiment",
        date="2026-05-01",
        status="complete",
    )
    
    assert entry.experiment_name == "test_exp"
    assert entry.dataset_name == "pcam"
    assert entry.dataset_subset_size == 1000
    assert entry.final_metrics["accuracy"] == 0.95
    assert len(entry.caveats) == 2


def test_benchmark_entry_to_dict():
    """Test BenchmarkEntry conversion to dict."""
    from dataclasses import asdict
    
    entry = BenchmarkEntry(
        experiment_name="test",
        dataset_name="pcam",
        dataset_subset_size=100,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={},
        artifact_paths={},
        caveats=[],
        notes="",
        date="2026-05-01",
        status="pending",
    )
    
    entry_dict = asdict(entry)
    
    assert isinstance(entry_dict, dict)
    assert entry_dict["experiment_name"] == "test"
    assert entry_dict["dataset_subset_size"] == 100


# ============================================================================
# BENCHMARK MANIFEST INITIALIZATION TESTS
# ============================================================================


def test_manifest_init_default_path(tmp_path, monkeypatch):
    """Test manifest initialization with default path."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    manifest = BenchmarkManifest()
    
    assert manifest.manifest_path == "benchmarks/manifest.jsonl"


def test_manifest_init_custom_path(tmp_path):
    """Test manifest initialization with custom path."""
    custom_path = tmp_path / "custom" / "manifest.jsonl"
    
    manifest = BenchmarkManifest(str(custom_path))
    
    assert manifest.manifest_path == str(custom_path)
    # Directory should be created
    assert custom_path.parent.exists()


def test_manifest_init_creates_directory(tmp_path):
    """Test manifest initialization creates parent directories."""
    manifest_path = tmp_path / "nested" / "dir" / "manifest.jsonl"
    
    manifest = BenchmarkManifest(str(manifest_path))
    
    assert manifest_path.parent.exists()


def test_manifest_init_no_directory_for_current_dir(tmp_path, monkeypatch):
    """Test manifest initialization with filename only (no directory)."""
    monkeypatch.chdir(tmp_path)
    
    # Should not raise error
    manifest = BenchmarkManifest("manifest.jsonl")
    
    assert manifest.manifest_path == "manifest.jsonl"


# ============================================================================
# ADD ENTRY TESTS
# ============================================================================


def test_add_entry_basic(tmp_path):
    """Test adding a single entry."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entry = BenchmarkEntry(
        experiment_name="exp1",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="config.yaml",
        train_command="python train.py",
        eval_command="python eval.py",
        final_metrics={"acc": 0.9},
        artifact_paths={"model": "model.pth"},
        caveats=[],
        notes="Test",
        date="2026-05-01",
        status="complete",
    )
    
    manifest.add_entry(entry)
    
    # Verify file exists and contains entry
    assert manifest_path.exists()
    content = manifest_path.read_text()
    assert "exp1" in content
    assert "pcam" in content


def test_add_entry_multiple(tmp_path):
    """Test adding multiple entries."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    for i in range(3):
        entry = BenchmarkEntry(
            experiment_name=f"exp{i}",
            dataset_name="pcam",
            dataset_subset_size=1000,
            config_path="config.yaml",
            train_command="train",
            eval_command="eval",
            final_metrics={},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-05-01",
            status="complete",
        )
        manifest.add_entry(entry)
    
    # Verify all entries present
    entries = manifest.read_all()
    assert len(entries) == 3
    assert entries[0].experiment_name == "exp0"
    assert entries[2].experiment_name == "exp2"


def test_add_entry_jsonl_format(tmp_path):
    """Test entries are written in JSON Lines format."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entry = BenchmarkEntry(
        experiment_name="test",
        dataset_name="pcam",
        dataset_subset_size=100,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={"acc": 0.9},
        artifact_paths={},
        caveats=[],
        notes="",
        date="2026-05-01",
        status="complete",
    )
    
    manifest.add_entry(entry)
    
    # Verify JSON Lines format (one JSON object per line)
    lines = manifest_path.read_text().strip().split("\n")
    assert len(lines) == 1
    
    # Verify line is valid JSON
    data = json.loads(lines[0])
    assert data["experiment_name"] == "test"


# ============================================================================
# UPDATE OR ADD ENTRY TESTS
# ============================================================================


def test_update_or_add_entry_new(tmp_path):
    """Test update_or_add_entry adds new entry."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entry = BenchmarkEntry(
        experiment_name="new_exp",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={},
        artifact_paths={},
        caveats=[],
        notes="",
        date="2026-05-01",
        status="complete",
    )
    
    updated = manifest.update_or_add_entry(entry)
    
    assert updated is False  # New entry added
    entries = manifest.read_all()
    assert len(entries) == 1
    assert entries[0].experiment_name == "new_exp"


def test_update_or_add_entry_existing(tmp_path):
    """Test update_or_add_entry updates existing entry."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    # Add initial entry
    entry1 = BenchmarkEntry(
        experiment_name="exp1",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={"acc": 0.8},
        artifact_paths={},
        caveats=[],
        notes="Initial",
        date="2026-05-01",
        status="complete",
    )
    manifest.add_entry(entry1)
    
    # Update with new metrics
    entry2 = BenchmarkEntry(
        experiment_name="exp1",  # Same name
        dataset_name="pcam",
        dataset_subset_size=2000,  # Different size
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={"acc": 0.9},  # Better metrics
        artifact_paths={},
        caveats=[],
        notes="Updated",
        date="2026-05-02",
        status="complete",
    )
    
    updated = manifest.update_or_add_entry(entry2)
    
    assert updated is True  # Existing entry updated
    entries = manifest.read_all()
    assert len(entries) == 1  # Still only one entry
    assert entries[0].final_metrics["acc"] == 0.9
    assert entries[0].notes == "Updated"
    assert entries[0].dataset_subset_size == 2000


def test_update_or_add_entry_preserves_others(tmp_path):
    """Test update_or_add_entry preserves other entries."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    # Add two entries
    for i in range(2):
        entry = BenchmarkEntry(
            experiment_name=f"exp{i}",
            dataset_name="pcam",
            dataset_subset_size=1000,
            config_path="config.yaml",
            train_command="train",
            eval_command="eval",
            final_metrics={},
            artifact_paths={},
            caveats=[],
            notes=f"Entry {i}",
            date="2026-05-01",
            status="complete",
        )
        manifest.add_entry(entry)
    
    # Update first entry
    updated_entry = BenchmarkEntry(
        experiment_name="exp0",
        dataset_name="pcam",
        dataset_subset_size=2000,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={"acc": 0.95},
        artifact_paths={},
        caveats=[],
        notes="Updated",
        date="2026-05-02",
        status="complete",
    )
    
    manifest.update_or_add_entry(updated_entry)
    
    entries = manifest.read_all()
    assert len(entries) == 2
    assert entries[0].notes == "Updated"
    assert entries[1].notes == "Entry 1"  # Preserved


# ============================================================================
# READ ALL TESTS
# ============================================================================


def test_read_all_empty_manifest(tmp_path):
    """Test reading from non-existent manifest."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entries = manifest.read_all()
    
    assert entries == []


def test_read_all_basic(tmp_path):
    """Test reading all entries."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    # Add entries
    for i in range(3):
        entry = BenchmarkEntry(
            experiment_name=f"exp{i}",
            dataset_name="pcam",
            dataset_subset_size=1000,
            config_path="config.yaml",
            train_command="train",
            eval_command="eval",
            final_metrics={"acc": 0.9 + i * 0.01},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-05-01",
            status="complete",
        )
        manifest.add_entry(entry)
    
    entries = manifest.read_all()
    
    assert len(entries) == 3
    assert all(isinstance(e, BenchmarkEntry) for e in entries)


def test_read_all_skips_corrupted_json(tmp_path):
    """Test reading skips corrupted JSON lines."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    # Write valid entry
    entry = BenchmarkEntry(
        experiment_name="valid",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={},
        artifact_paths={},
        caveats=[],
        notes="",
        date="2026-05-01",
        status="complete",
    )
    manifest.add_entry(entry)
    
    # Append corrupted line
    with open(manifest_path, "a") as f:
        f.write("{ invalid json }\n")
    
    entries = manifest.read_all()
    
    # Should only get valid entry
    assert len(entries) == 1
    assert entries[0].experiment_name == "valid"


def test_read_all_skips_invalid_schema(tmp_path):
    """Test reading skips entries with invalid schema."""
    manifest_path = tmp_path / "manifest.jsonl"
    
    # Write valid entry
    valid_entry = {
        "experiment_name": "valid",
        "dataset_name": "pcam",
        "dataset_subset_size": 1000,
        "config_path": "config.yaml",
        "train_command": "train",
        "eval_command": "eval",
        "final_metrics": {},
        "artifact_paths": {},
        "caveats": [],
        "notes": "",
        "date": "2026-05-01",
        "status": "complete",
    }
    
    # Write invalid entry (missing required field)
    invalid_entry = {
        "experiment_name": "invalid",
        # Missing other required fields
    }
    
    with open(manifest_path, "w") as f:
        json.dump(valid_entry, f)
        f.write("\n")
        json.dump(invalid_entry, f)
        f.write("\n")
    
    manifest = BenchmarkManifest(str(manifest_path))
    entries = manifest.read_all()
    
    # Should only get valid entry
    assert len(entries) == 1
    assert entries[0].experiment_name == "valid"


# ============================================================================
# FIND BY EXPERIMENT TESTS
# ============================================================================


def test_find_by_experiment_found(tmp_path):
    """Test finding existing experiment."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    # Add entries
    for i in range(3):
        entry = BenchmarkEntry(
            experiment_name=f"exp{i}",
            dataset_name="pcam",
            dataset_subset_size=1000,
            config_path="config.yaml",
            train_command="train",
            eval_command="eval",
            final_metrics={"acc": 0.9 + i * 0.01},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-05-01",
            status="complete",
        )
        manifest.add_entry(entry)
    
    found = manifest.find_by_experiment("exp1")
    
    assert found is not None
    assert found.experiment_name == "exp1"
    assert found.final_metrics["acc"] == 0.91


def test_find_by_experiment_not_found(tmp_path):
    """Test finding non-existent experiment."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entry = BenchmarkEntry(
        experiment_name="exp1",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={},
        artifact_paths={},
        caveats=[],
        notes="",
        date="2026-05-01",
        status="complete",
    )
    manifest.add_entry(entry)
    
    found = manifest.find_by_experiment("nonexistent")
    
    assert found is None


# ============================================================================
# MARKDOWN EXPORT TESTS
# ============================================================================


def test_to_markdown_basic(tmp_path):
    """Test Markdown export."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entry = BenchmarkEntry(
        experiment_name="test_exp",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="configs/test.yaml",
        train_command="python train.py --config configs/test.yaml",
        eval_command="python eval.py --checkpoint model.pth",
        final_metrics={"accuracy": 0.95, "auc": 0.98},
        artifact_paths={"model": "checkpoints/model.pth"},
        caveats=["Small dataset", "No validation split"],
        notes="Test experiment",
        date="2026-05-01",
        status="complete",
    )
    manifest.add_entry(entry)
    
    output_path = tmp_path / "summary.md"
    manifest.to_markdown(str(output_path))
    
    # Verify Markdown file created
    assert output_path.exists()
    
    content = output_path.read_text()
    
    # Verify content
    assert "# Benchmark Results Manifest" in content
    assert "## test_exp" in content
    assert "**Dataset**: pcam" in content
    assert "**Subset Size**: 1000" in content
    assert "**Status**: complete" in content
    assert "**accuracy**: 0.95" in content
    assert "**auc**: 0.98" in content
    assert "python train.py" in content
    assert "python eval.py" in content
    assert "Small dataset" in content
    assert "No validation split" in content


def test_to_markdown_multiple_entries(tmp_path):
    """Test Markdown export with multiple entries."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    for i in range(2):
        entry = BenchmarkEntry(
            experiment_name=f"exp{i}",
            dataset_name="pcam",
            dataset_subset_size=1000 * (i + 1),
            config_path="config.yaml",
            train_command="train",
            eval_command="eval",
            final_metrics={"acc": 0.9 + i * 0.05},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-05-01",
            status="complete",
        )
        manifest.add_entry(entry)
    
    output_path = tmp_path / "summary.md"
    manifest.to_markdown(str(output_path))
    
    content = output_path.read_text()
    
    # Verify both entries present
    assert "## exp0" in content
    assert "## exp1" in content
    assert "**Subset Size**: 1000" in content
    assert "**Subset Size**: 2000" in content


def test_to_markdown_empty_caveats(tmp_path):
    """Test Markdown export with no caveats."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = BenchmarkManifest(str(manifest_path))
    
    entry = BenchmarkEntry(
        experiment_name="test",
        dataset_name="pcam",
        dataset_subset_size=1000,
        config_path="config.yaml",
        train_command="train",
        eval_command="eval",
        final_metrics={},
        artifact_paths={},
        caveats=[],  # Empty
        notes="",
        date="2026-05-01",
        status="complete",
    )
    manifest.add_entry(entry)
    
    output_path = tmp_path / "summary.md"
    manifest.to_markdown(str(output_path))
    
    content = output_path.read_text()
    
    # Should not have Caveats section
    assert "### Caveats" not in content
