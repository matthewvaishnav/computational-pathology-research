"""
Regression tests for benchmark manifest utilities.

Tests focused on:
- BenchmarkEntry dataclass behavior
- BenchmarkManifest read/write operations
- Markdown generation
- Edge cases (missing files, empty manifests)
"""

import json
import os
from pathlib import Path

import pytest

from src.utils.benchmark_manifest import BenchmarkEntry, BenchmarkManifest


class TestBenchmarkEntry:
    """Test BenchmarkEntry dataclass creation and serialization."""

    @pytest.fixture
    def sample_entry(self):
        """Create a sample benchmark entry."""
        return BenchmarkEntry(
            experiment_name="test_experiment",
            dataset_name="TestDataset",
            dataset_subset_size=1000,
            config_path="configs/test.yaml",
            train_command="python train.py --config configs/test.yaml",
            eval_command="python eval.py --checkpoint best.pth",
            final_metrics={
                "accuracy": 0.95,
                "auc": 0.97,
                "f1": 0.94,
            },
            artifact_paths={
                "checkpoints": ["checkpoints/best.pth"],
                "logs": ["logs/training.log"],
            },
            caveats=["Synthetic data", "Small sample"],
            notes="Test run for validation",
            date="2026-04-07",
            status="COMPLETE",
        )

    def test_entry_creation(self, sample_entry):
        """BenchmarkEntry can be created with all fields."""
        assert sample_entry.experiment_name == "test_experiment"
        assert sample_entry.dataset_name == "TestDataset"
        assert sample_entry.dataset_subset_size == 1000
        assert sample_entry.status == "COMPLETE"

    def test_entry_serialization(self, sample_entry):
        """BenchmarkEntry can be serialized to dict."""
        from dataclasses import asdict

        data = asdict(sample_entry)
        assert data["experiment_name"] == "test_experiment"
        assert data["final_metrics"]["accuracy"] == 0.95
        assert len(data["caveats"]) == 2

    def test_entry_deserialization(self, sample_entry):
        """BenchmarkEntry can be recreated from dict."""
        from dataclasses import asdict

        data = asdict(sample_entry)
        restored = BenchmarkEntry(**data)

        assert restored.experiment_name == sample_entry.experiment_name
        assert restored.final_metrics == sample_entry.final_metrics
        assert restored.caveats == sample_entry.caveats


class TestBenchmarkManifestEmpty:
    """Test BenchmarkManifest behavior with missing/empty files."""

    def test_missing_manifest_returns_empty_list(self, tmp_path):
        """read_all() returns empty list when manifest doesn't exist."""
        manifest_path = tmp_path / "nonexistent" / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))

        entries = manifest.read_all()
        assert entries == []

    def test_empty_manifest_returns_empty_list(self, tmp_path):
        """read_all() returns empty list for empty manifest file."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text("")

        manifest = BenchmarkManifest(str(manifest_path))
        entries = manifest.read_all()
        assert entries == []

    def test_whitespace_only_manifest(self, tmp_path):
        """read_all() handles whitespace-only manifest file."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text("   \n\n   ")

        manifest = BenchmarkManifest(str(manifest_path))
        entries = manifest.read_all()
        assert entries == []


class TestBenchmarkManifestAddRead:
    """Test adding and reading entries from manifest."""

    @pytest.fixture
    def manifest(self, tmp_path):
        """Create a temporary manifest."""
        manifest_path = tmp_path / "manifest.jsonl"
        return BenchmarkManifest(str(manifest_path))

    @pytest.fixture
    def sample_entry(self):
        """Create a sample benchmark entry."""
        return BenchmarkEntry(
            experiment_name="test_experiment",
            dataset_name="TestDataset",
            dataset_subset_size=1000,
            config_path="configs/test.yaml",
            train_command="python train.py",
            eval_command="python eval.py",
            final_metrics={"accuracy": 0.95},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-04-07",
            status="COMPLETE",
        )

    def test_add_single_entry(self, manifest, sample_entry):
        """Can add a single entry to manifest."""
        manifest.add_entry(sample_entry)

        entries = manifest.read_all()
        assert len(entries) == 1
        assert entries[0].experiment_name == "test_experiment"

    def test_add_multiple_entries(self, manifest, sample_entry):
        """Can add multiple entries to manifest."""
        entry1 = sample_entry
        entry2 = BenchmarkEntry(
            experiment_name="second_experiment",
            dataset_name="OtherDataset",
            dataset_subset_size=500,
            config_path="configs/other.yaml",
            train_command="python train.py",
            eval_command="python eval.py",
            final_metrics={"accuracy": 0.90},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-04-08",
            status="COMPLETE",
        )

        manifest.add_entry(entry1)
        manifest.add_entry(entry2)

        entries = manifest.read_all()
        assert len(entries) == 2
        assert entries[0].experiment_name == "test_experiment"
        assert entries[1].experiment_name == "second_experiment"

    def test_persistence_across_instances(self, tmp_path, sample_entry):
        """Entries persist when read by new BenchmarkManifest instance."""
        manifest_path = tmp_path / "manifest.jsonl"

        # Add entry with first instance
        manifest1 = BenchmarkManifest(str(manifest_path))
        manifest1.add_entry(sample_entry)

        # Read with second instance
        manifest2 = BenchmarkManifest(str(manifest_path))
        entries = manifest2.read_all()

        assert len(entries) == 1
        assert entries[0].experiment_name == "test_experiment"

    def test_json_lines_format(self, manifest, sample_entry):
        """Entries are stored in JSON Lines format (one per line)."""
        manifest.add_entry(sample_entry)

        # Read raw file content
        with open(manifest.manifest_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        # Each line should be valid JSON
        data = json.loads(lines[0])
        assert data["experiment_name"] == "test_experiment"


class TestFindByExperiment:
    """Test find_by_experiment() method."""

    @pytest.fixture
    def populated_manifest(self, tmp_path):
        """Create a manifest with multiple entries."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))

        entries = [
            BenchmarkEntry(
                experiment_name="pcam_baseline",
                dataset_name="PCam",
                dataset_subset_size=1000,
                config_path="pcam.yaml",
                train_command="train.py",
                eval_command="eval.py",
                final_metrics={},
                artifact_paths={},
                caveats=[],
                notes="",
                date="2026-04-07",
                status="COMPLETE",
            ),
            BenchmarkEntry(
                experiment_name="resnet50_variant",
                dataset_name="PCam",
                dataset_subset_size=1000,
                config_path="resnet50.yaml",
                train_command="train.py",
                eval_command="eval.py",
                final_metrics={},
                artifact_paths={},
                caveats=[],
                notes="",
                date="2026-04-08",
                status="COMPLETE",
            ),
            BenchmarkEntry(
                experiment_name="simple_head",
                dataset_name="PCam",
                dataset_subset_size=500,
                config_path="simple.yaml",
                train_command="train.py",
                eval_command="eval.py",
                final_metrics={},
                artifact_paths={},
                caveats=[],
                notes="",
                date="2026-04-09",
                status="FAILED",
            ),
        ]

        for entry in entries:
            manifest.add_entry(entry)

        return manifest

    def test_find_existing_experiment(self, populated_manifest):
        """Can find existing experiment by name."""
        entry = populated_manifest.find_by_experiment("resnet50_variant")

        assert entry is not None
        assert entry.experiment_name == "resnet50_variant"
        assert entry.status == "COMPLETE"

    def test_find_first_match(self, populated_manifest):
        """Returns first match when multiple entries exist."""
        entry = populated_manifest.find_by_experiment("pcam_baseline")

        assert entry is not None
        assert entry.experiment_name == "pcam_baseline"

    def test_find_nonexistent_returns_none(self, populated_manifest):
        """Returns None when experiment not found."""
        entry = populated_manifest.find_by_experiment("nonexistent")

        assert entry is None

    def test_find_in_empty_manifest(self, tmp_path):
        """Returns None when manifest is empty."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))

        entry = manifest.find_by_experiment("anything")
        assert entry is None


class TestToMarkdown:
    """Test to_markdown() summary generation."""

    @pytest.fixture
    def sample_entry(self):
        """Create a sample entry with rich data."""
        return BenchmarkEntry(
            experiment_name="pcam_baseline",
            dataset_name="PatchCamelyon",
            dataset_subset_size=700,
            config_path="configs/pcam.yaml",
            train_command="python train.py --config configs/pcam.yaml",
            eval_command="python eval.py --checkpoint best.pth",
            final_metrics={
                "accuracy": 0.94,
                "auc": 0.97,
                "f1": 0.93,
            },
            artifact_paths={
                "checkpoints": ["checkpoints/best.pth"],
            },
            caveats=["Synthetic data", "Small sample"],
            notes="Test run",
            date="2026-04-07",
            status="COMPLETE",
        )

    def test_markdown_generation(self, tmp_path, sample_entry):
        """Markdown file is generated with correct structure."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))
        manifest.add_entry(sample_entry)

        output_path = tmp_path / "summary.md"
        manifest.to_markdown(str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        # Check header
        assert "# Benchmark Results Manifest" in content

        # Check experiment section
        assert "## pcam_baseline" in content

        # Check metadata fields
        assert "**Dataset**: PatchCamelyon" in content
        assert "**Subset Size**: 700" in content
        assert "**Status**: COMPLETE" in content

        # Check metrics section
        assert "### Metrics" in content
        assert "**accuracy**: 0.94" in content
        assert "**auc**: 0.97" in content

        # Check commands section
        assert "### Commands" in content
        assert "python train.py --config configs/pcam.yaml" in content
        assert "python eval.py --checkpoint best.pth" in content

        # Check caveats section
        assert "### Caveats" in content
        assert "Synthetic data" in content

    def test_markdown_empty_metrics(self, tmp_path):
        """Markdown handles empty metrics gracefully."""
        entry = BenchmarkEntry(
            experiment_name="empty_test",
            dataset_name="Test",
            dataset_subset_size=100,
            config_path="test.yaml",
            train_command="train.py",
            eval_command="eval.py",
            final_metrics={},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-04-07",
            status="PENDING",
        )

        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))
        manifest.add_entry(entry)

        output_path = tmp_path / "summary.md"
        manifest.to_markdown(str(output_path))

        content = output_path.read_text()
        assert "### Metrics" in content
        assert "## empty_test" in content

    def test_markdown_no_caveats(self, tmp_path):
        """Markdown omits caveats section when empty."""
        entry = BenchmarkEntry(
            experiment_name="no_caveats",
            dataset_name="Test",
            dataset_subset_size=100,
            config_path="test.yaml",
            train_command="train.py",
            eval_command="eval.py",
            final_metrics={"accuracy": 0.9},
            artifact_paths={},
            caveats=[],  # Empty caveats
            notes="",
            date="2026-04-07",
            status="COMPLETE",
        )

        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))
        manifest.add_entry(entry)

        output_path = tmp_path / "summary.md"
        manifest.to_markdown(str(output_path))

        content = output_path.read_text()
        # Should not have caveats section
        assert "### Caveats" not in content

    def test_markdown_multiple_entries(self, tmp_path):
        """Markdown includes all entries in order."""
        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))

        for i in range(3):
            entry = BenchmarkEntry(
                experiment_name=f"exp_{i}",
                dataset_name="Test",
                dataset_subset_size=100,
                config_path="test.yaml",
                train_command="train.py",
                eval_command="eval.py",
                final_metrics={"id": i},
                artifact_paths={},
                caveats=[],
                notes="",
                date="2026-04-07",
                status="COMPLETE",
            )
            manifest.add_entry(entry)

        output_path = tmp_path / "summary.md"
        manifest.to_markdown(str(output_path))

        content = output_path.read_text()
        assert "## exp_0" in content
        assert "## exp_1" in content
        assert "## exp_2" in content


class TestManifestEdgeCases:
    """Test edge cases and error conditions."""

    def test_directory_creation(self, tmp_path):
        """BenchmarkManifest creates parent directories if needed."""
        manifest_path = tmp_path / "nested" / "deep" / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))

        # Directory should be created
        assert manifest_path.parent.exists()

    def test_corrupted_line_handling(self, tmp_path):
        """read_all() skips corrupted JSON lines gracefully."""
        manifest_path = tmp_path / "manifest.jsonl"
        # First line is valid JSON but not a valid BenchmarkEntry
        # Second line is invalid JSON
        manifest_path.write_text('{"valid": "json"}\n{invalid json}\n')

        manifest = BenchmarkManifest(str(manifest_path))

        # Should skip corrupted lines and return empty list
        entries = manifest.read_all()
        assert entries == []

    def test_partial_corruption_with_valid_entries(self, tmp_path):
        """read_all() skips corrupted lines but keeps valid entries."""
        from dataclasses import asdict

        manifest_path = tmp_path / "manifest.jsonl"

        # Create a valid entry
        valid_entry = BenchmarkEntry(
            experiment_name="valid_exp",
            dataset_name="Test",
            dataset_subset_size=100,
            config_path="test.yaml",
            train_command="train.py",
            eval_command="eval.py",
            final_metrics={},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-04-07",
            status="COMPLETE",
        )

        # Write valid entry, then corrupted line
        with open(manifest_path, "w") as f:
            f.write(json.dumps(asdict(valid_entry)) + "\n")
            f.write('{"invalid": "entry"}\n')  # Valid JSON but invalid entry
            f.write("{not valid json}\n")  # Invalid JSON

        manifest = BenchmarkManifest(str(manifest_path))
        entries = manifest.read_all()

        # Should have only the valid entry
        assert len(entries) == 1
        assert entries[0].experiment_name == "valid_exp"

    def test_unicode_in_fields(self, tmp_path):
        """Handles unicode characters in entry fields."""
        entry = BenchmarkEntry(
            experiment_name="test_日本語",
            dataset_name="Dataset with 🎉 emoji",
            dataset_subset_size=100,
            config_path="config.yaml",
            train_command="train.py",
            eval_command="eval.py",
            final_metrics={"metric": "value with ñ"},
            artifact_paths={},
            caveats=["Caveat with ü"],
            notes="Note with é",
            date="2026-04-07",
            status="COMPLETE",
        )

        manifest_path = tmp_path / "manifest.jsonl"
        manifest = BenchmarkManifest(str(manifest_path))
        manifest.add_entry(entry)

        entries = manifest.read_all()
        assert len(entries) == 1
        assert entries[0].experiment_name == "test_日本語"
        assert entries[0].dataset_name == "Dataset with 🎉 emoji"
