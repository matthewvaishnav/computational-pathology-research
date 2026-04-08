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


class TestManifestIsolation:
    """Regression tests to prevent test pollution of committed manifest."""

    def test_default_manifest_path_constant(self):
        """DEFAULT_MANIFEST_PATH points to committed manifest location."""
        assert BenchmarkManifest.DEFAULT_MANIFEST_PATH == "benchmarks/manifest.jsonl"

    def test_manifest_path_injectability(self, tmp_path):
        """BenchmarkManifest accepts injectable manifest path."""
        custom_path = tmp_path / "custom_manifest.jsonl"
        manifest = BenchmarkManifest(manifest_path=str(custom_path))

        # Verify the path was set correctly
        assert manifest.manifest_path == str(custom_path)

    def test_default_path_when_none_provided(self):
        """BenchmarkManifest uses default path when manifest_path is None."""
        manifest = BenchmarkManifest(manifest_path=None)
        assert manifest.manifest_path == BenchmarkManifest.DEFAULT_MANIFEST_PATH

    def test_test_entries_dont_pollute_committed_manifest(self, tmp_path):
        """Test entries written to temp path do not appear in default manifest."""
        # Create a test entry
        entry = BenchmarkEntry(
            experiment_name="test_isolation_entry",
            dataset_name="TestDataset",
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

        # Write to temp manifest
        temp_manifest = BenchmarkManifest(manifest_path=str(tmp_path / "temp.jsonl"))
        temp_manifest.add_entry(entry)

        # Verify default manifest does NOT contain this entry
        default_manifest = BenchmarkManifest()
        default_entries = default_manifest.read_all()

        # None of the default entries should have our test experiment name
        entry_names = [e.experiment_name for e in default_entries]
        assert "test_isolation_entry" not in entry_names, \
            "Test entry polluted the committed benchmark manifest!"


class TestUpdateOrAddEntry:
    """Test update_or_add_entry() duplicate prevention."""

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

    def test_adds_new_entry_when_empty(self, manifest, sample_entry):
        """update_or_add_entry adds entry when manifest is empty."""
        result = manifest.update_or_add_entry(sample_entry)

        assert result is False  # Indicates new entry was added
        entries = manifest.read_all()
        assert len(entries) == 1
        assert entries[0].experiment_name == "test_experiment"

    def test_updates_existing_entry(self, manifest, sample_entry):
        """update_or_add_entry updates existing entry instead of duplicating."""
        # Add initial entry
        manifest.add_entry(sample_entry)

        # Create updated version with different metrics
        updated_entry = BenchmarkEntry(
            experiment_name="test_experiment",  # Same name
            dataset_name="TestDataset",
            dataset_subset_size=1000,
            config_path="configs/test.yaml",
            train_command="python train.py",
            eval_command="python eval.py",
            final_metrics={"accuracy": 0.97},  # Updated metric
            artifact_paths={"new": "path"},
            caveats=["updated"],
            notes="updated notes",
            date="2026-04-08",
            status="COMPLETE",
        )

        result = manifest.update_or_add_entry(updated_entry)

        assert result is True  # Indicates existing entry was updated
        entries = manifest.read_all()
        assert len(entries) == 1  # No duplicate!
        assert entries[0].final_metrics["accuracy"] == 0.97
        assert entries[0].notes == "updated notes"

    def test_prevents_multiple_duplicates(self, manifest, sample_entry):
        """Multiple updates keep only one entry."""
        # Add initial entry
        manifest.add_entry(sample_entry)

        # Update 3 times
        for i in range(3):
            updated = BenchmarkEntry(
                experiment_name="test_experiment",
                dataset_name="TestDataset",
                dataset_subset_size=1000,
                config_path="configs/test.yaml",
                train_command="python train.py",
                eval_command="python eval.py",
                final_metrics={"accuracy": 0.90 + i * 0.01},
                artifact_paths={},
                caveats=[],
                notes=f"update {i}",
                date="2026-04-07",
                status="COMPLETE",
            )
            manifest.update_or_add_entry(updated)

        entries = manifest.read_all()
        assert len(entries) == 1  # Still only one entry
        assert entries[0].notes == "update 2"  # Last update wins

    def test_handles_multiple_different_experiments(self, manifest):
        """update_or_add_entry handles multiple distinct experiments."""
        entries = []
        for i in range(3):
            entry = BenchmarkEntry(
                experiment_name=f"experiment_{i}",
                dataset_name="TestDataset",
                dataset_subset_size=1000,
                config_path="configs/test.yaml",
                train_command="python train.py",
                eval_command="python eval.py",
                final_metrics={"id": i},
                artifact_paths={},
                caveats=[],
                notes="",
                date="2026-04-07",
                status="COMPLETE",
            )
            entries.append(entry)
            manifest.update_or_add_entry(entry)

        # Should have 3 entries
        all_entries = manifest.read_all()
        assert len(all_entries) == 3

        # Update middle one
        updated = BenchmarkEntry(
            experiment_name="experiment_1",  # Same name as middle
            dataset_name="UpdatedDataset",
            dataset_subset_size=1000,
            config_path="configs/test.yaml",
            train_command="python train.py",
            eval_command="python eval.py",
            final_metrics={"id": 99},
            artifact_paths={},
            caveats=[],
            notes="",
            date="2026-04-07",
            status="COMPLETE",
        )
        result = manifest.update_or_add_entry(updated)

        assert result is True  # Updated existing
        all_entries = manifest.read_all()
        assert len(all_entries) == 3  # Still 3, no new entry added

        # Verify the update
        exp1 = [e for e in all_entries if e.experiment_name == "experiment_1"][0]
        assert exp1.dataset_name == "UpdatedDataset"
        assert exp1.final_metrics["id"] == 99
