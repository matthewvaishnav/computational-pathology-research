"""
Benchmark manifest utilities for tracking experiment results.

Lightweight helper for reading/writing benchmark entries to a JSON Lines manifest.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """Single benchmark run entry."""

    experiment_name: str
    dataset_name: str
    dataset_subset_size: int
    config_path: str
    train_command: str
    eval_command: str
    final_metrics: dict
    artifact_paths: dict
    caveats: list[str]
    notes: str
    date: str
    status: str


class BenchmarkManifest:
    """Manages benchmark entries in a JSON Lines manifest file."""

    # Default path for the committed benchmark manifest
    DEFAULT_MANIFEST_PATH = "benchmarks/manifest.jsonl"

    def __init__(self, manifest_path: str = None):
        """Initialize manifest.

        Args:
            manifest_path: Path to manifest file. If None, uses DEFAULT_MANIFEST_PATH.
        """
        if manifest_path is None:
            manifest_path = self.DEFAULT_MANIFEST_PATH
        self.manifest_path = manifest_path
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    def add_entry(self, entry: BenchmarkEntry) -> None:
        """Append a benchmark entry to the manifest."""
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            # Write compact single-line JSON for JSON Lines format
            json.dump(asdict(entry), f, separators=(",", ":"))
            f.write("\n")

    def read_all(self) -> list[BenchmarkEntry]:
        """Read all benchmark entries from the manifest.

        Skips corrupted lines and logs warnings for invalid entries.
        """
        entries = []
        if not os.path.exists(self.manifest_path):
            return entries

        with open(self.manifest_path, encoding="utf-8") as f:
            content = f.read()

        # Handle both single-line JSONL and formatted multi-line JSON
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]

        for line_num, line in enumerate(lines, 1):
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(BenchmarkEntry(**data))
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                logger.warning(f"Skipping corrupted line {line_num} in manifest")
                continue
            except TypeError as e:
                # Skip lines that don't match BenchmarkEntry schema
                logger.warning(f"Skipping invalid entry at line {line_num}: {e}")
                continue

        return entries

    def find_by_experiment(self, name: str) -> Optional[BenchmarkEntry]:
        """Find a benchmark entry by experiment name."""
        for entry in self.read_all():
            if entry.experiment_name == name:
                return entry
        return None

    def to_markdown(self, output_path: str) -> None:
        """Generate a Markdown summary of all benchmarks."""
        entries = self.read_all()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Results Manifest\n\n")
            for entry in entries:
                f.write(f"## {entry.experiment_name}\n\n")
                f.write(f"- **Dataset**: {entry.dataset_name}\n")
                f.write(f"- **Subset Size**: {entry.dataset_subset_size}\n")
                f.write(f"- **Status**: {entry.status}\n")
                f.write(f"- **Date**: {entry.date}\n\n")
                f.write("### Metrics\n")
                for key, value in entry.final_metrics.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write(f"\n### Commands\n\n**Train**:\n```bash\n{entry.train_command}\n```\n\n")
                f.write(f"**Eval**:\n```bash\n{entry.eval_command}\n```\n\n")
                if entry.caveats:
                    f.write("### Caveats\n")
                    for caveat in entry.caveats:
                        f.write(f"- {caveat}\n")
                    f.write("\n")
                f.write("---\n\n")


if __name__ == "__main__":
    # Example: Read and print current manifest
    manifest = BenchmarkManifest()
    entries = manifest.read_all()
    print(f"Found {len(entries)} benchmark entries")
    for e in entries:
        print(f"  - {e.experiment_name}: {e.status}")
