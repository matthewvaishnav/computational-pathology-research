from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.camelyon17.audit_camelyon17_wilds import (
    metadata_dataframe,
    pick_client_column,
    split_dataframe,
    summarize_clients,
    write_markdown_report,
)


@dataclass
class FakeDataset:
    metadata_fields = ["hospital", "slide"]
    split_dict = {"train": 0, "val": 1, "test": 2}
    split_array = [0, 0, 1, 2]

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        ys = [0, 1, 1, 0]
        metadata = [[0, 100], [0, 101], [1, 102], [2, 103]][idx]
        return None, ys[idx], metadata


def test_metadata_dataframe_extracts_labels_and_metadata() -> None:
    df = metadata_dataframe(FakeDataset())

    assert list(df.columns) == ["index", "y", "hospital", "slide"]
    assert df.shape == (4, 4)
    assert df["hospital"].tolist() == [0, 0, 1, 2]


def test_split_dataframe_maps_split_ids_to_names() -> None:
    splits = split_dataframe(FakeDataset())

    assert splits["split"].tolist() == ["train", "train", "val", "test"]


def test_pick_client_column_prefers_hospital() -> None:
    df = pd.DataFrame(
        {
            "index": [0, 1, 2],
            "y": [0, 1, 0],
            "hospital": [0, 0, 1],
            "slide": [10, 11, 12],
        }
    )

    assert pick_client_column(df) == "hospital"


def test_summarize_clients_counts_classes_and_splits() -> None:
    metadata = metadata_dataframe(FakeDataset())
    splits = split_dataframe(FakeDataset())
    df = metadata.merge(splits, on="index", how="left")

    summary = summarize_clients(df, "hospital")

    assert summary.iloc[0]["client_id"] == 0
    assert summary.iloc[0]["total_n"] == 2
    assert set(summary["client_id"].tolist()) == {0, 1, 2}


def test_write_markdown_report(tmp_path: Path) -> None:
    metadata = metadata_dataframe(FakeDataset())
    splits = split_dataframe(FakeDataset())
    df = metadata.merge(splits, on="index", how="left")
    summary = summarize_clients(df, "hospital")
    out = tmp_path / "audit.md"

    write_markdown_report(out, FakeDataset(), df, "hospital", summary)

    text = out.read_text(encoding="utf-8")
    assert "Camelyon17/WILDS dataset audit" in text
    assert "Selected FL client column: `hospital`" in text
    assert "Client summary" in text
