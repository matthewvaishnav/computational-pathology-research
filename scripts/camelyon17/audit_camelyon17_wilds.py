#!/usr/bin/env python3
"""Audit Camelyon17/WILDS metadata for external FL validation.

This is the first Boss-1 external-validation script. It intentionally does not
train a model. It answers the prerequisite question: what hospital/domain
structure is available, and how should it map to simulated FL clients?

Examples
--------
Run without downloading, using an existing WILDS data directory:

    python scripts/camelyon17/audit_camelyon17_wilds.py --root data/wilds --download false

Allow WILDS to download the dataset:

    python scripts/camelyon17/audit_camelyon17_wilds.py --root data/wilds --download true

Write artifacts somewhere else:

    python scripts/camelyon17/audit_camelyon17_wilds.py --out-dir results/camelyon17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_bool(value: str | bool) -> bool:
    """Parse a command-line bool without surprising argparse behavior."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _to_plain(value: Any) -> Any:
    """Convert tensors / numpy scalars / lists into JSON-serializable values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def load_wilds_camelyon17(root: Path, download: bool):
    """Load Camelyon17 from WILDS with a helpful error if WILDS is missing."""
    try:
        from wilds import get_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise SystemExit(
            "WILDS is not installed. Install with: python -m pip install wilds"
        ) from exc

    return get_dataset(dataset="camelyon17", root_dir=str(root), download=download)


def metadata_dataframe(dataset: Any) -> pd.DataFrame:
    """Return one row per example with y plus WILDS metadata columns."""
    rows: list[dict[str, Any]] = []
    metadata_fields = list(getattr(dataset, "metadata_fields", []))

    for idx in range(len(dataset)):
        _, y, metadata = dataset[idx]
        metadata_values = _to_plain(metadata)
        if not isinstance(metadata_values, list):
            metadata_values = [metadata_values]

        row = {"index": idx, "y": _to_plain(y)}
        for field, value in zip(metadata_fields, metadata_values):
            row[str(field)] = value
        rows.append(row)

    return pd.DataFrame(rows)


def split_dataframe(dataset: Any) -> pd.DataFrame:
    """Return split membership using WILDS split arrays when available."""
    rows: list[dict[str, Any]] = []
    split_dict = getattr(dataset, "split_dict", {})
    split_array = getattr(dataset, "split_array", None)

    if split_array is None:
        return pd.DataFrame(columns=["index", "split"])

    inverse = {int(value): str(name) for name, value in split_dict.items()}
    split_values = _to_plain(split_array)
    for idx, split_id in enumerate(split_values):
        rows.append({"index": idx, "split": inverse.get(int(split_id), str(split_id))})
    return pd.DataFrame(rows)


def pick_client_column(df: pd.DataFrame) -> str:
    """Pick the most likely hospital/domain column in Camelyon17 metadata."""
    candidates = ["hospital", "center", "site", "domain"]
    lower_to_actual = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]

    # Fallback: choose the non-label metadata column with the lowest cardinality
    # above one. This avoids silently assuming sample index is the client.
    usable = [c for c in df.columns if c not in {"index", "y", "split"}]
    if not usable:
        raise SystemExit("Could not identify any metadata column to use as FL client_id")

    cardinalities = [(c, df[c].nunique(dropna=False)) for c in usable]
    cardinalities = [item for item in cardinalities if item[1] > 1]
    if not cardinalities:
        raise SystemExit("Metadata exists, but no metadata column has more than one value")
    return min(cardinalities, key=lambda item: item[1])[0]


def summarize_clients(df: pd.DataFrame, client_column: str) -> pd.DataFrame:
    """Build split/class/client counts for FL client construction."""
    group_cols = [client_column]
    if "split" in df.columns:
        group_cols.append("split")
    group_cols.append("y")

    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="n")
    pivot = counts.pivot_table(
        index=client_column,
        columns=[c for c in group_cols if c != client_column],
        values="n",
        fill_value=0,
        aggfunc="sum",
    )
    pivot.columns = ["__".join(map(str, col if isinstance(col, tuple) else (col,))) for col in pivot.columns]
    pivot = pivot.reset_index().rename(columns={client_column: "client_id"})

    totals = df.groupby(client_column, dropna=False).size().reset_index(name="total_n")
    totals = totals.rename(columns={client_column: "client_id"})
    return totals.merge(pivot, on="client_id", how="left").sort_values("total_n", ascending=False)


def write_markdown_report(
    out_path: Path,
    dataset: Any,
    df: pd.DataFrame,
    client_column: str,
    client_summary: pd.DataFrame,
) -> None:
    """Write a short audit report for the paper/repo."""
    split_counts = df["split"].value_counts(dropna=False).to_dict() if "split" in df else {}
    class_counts = df["y"].value_counts(dropna=False).sort_index().to_dict()
    metadata_fields = list(getattr(dataset, "metadata_fields", []))

    lines = [
        "# Camelyon17/WILDS dataset audit",
        "",
        "## Purpose",
        "",
        "Prepare Camelyon17 as the first natural multi-site external validation target for the dominant-site detector-switch hypothesis.",
        "",
        "## Dataset metadata",
        "",
        f"- Total examples: {len(df):,}",
        f"- Metadata fields: {metadata_fields}",
        f"- Selected FL client column: `{client_column}`",
        "",
        "## Split counts",
        "",
        "```json",
        json.dumps(split_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Class counts",
        "",
        "```json",
        json.dumps({str(k): int(v) for k, v in class_counts.items()}, indent=2, sort_keys=True),
        "```",
        "",
        "## Client summary",
        "",
        client_summary.to_markdown(index=False),
        "",
        "## Next decision",
        "",
        "Use the selected client column as the simulated FL `client_id`, then compare FedAvg, equal-client weighting, FedProx, and detector-switch logic on worst-site and global metrics.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/camelyon17"))
    parser.add_argument("--download", type=parse_bool, default=False)
    args = parser.parse_args()

    dataset = load_wilds_camelyon17(args.root, args.download)
    metadata = metadata_dataframe(dataset)
    splits = split_dataframe(dataset)
    df = metadata.merge(splits, on="index", how="left") if not splits.empty else metadata

    client_column = pick_client_column(df)
    client_summary = summarize_clients(df, client_column)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "metadata_audit.csv", index=False)
    client_summary.to_csv(args.out_dir / "client_audit.csv", index=False)
    write_markdown_report(
        args.out_dir / "camelyon17_dataset_audit.md",
        dataset,
        df,
        client_column,
        client_summary,
    )

    print(f"Wrote {args.out_dir / 'metadata_audit.csv'}")
    print(f"Wrote {args.out_dir / 'client_audit.csv'}")
    print(f"Wrote {args.out_dir / 'camelyon17_dataset_audit.md'}")
    print(f"Selected client column: {client_column}")


if __name__ == "__main__":
    main()
