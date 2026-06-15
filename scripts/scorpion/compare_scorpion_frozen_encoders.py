#!/usr/bin/env python3
"""Compare frozen SCORPION encoder audits without touching the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    extraction = payload.get("extraction_metadata", {})
    probe = payload.get("scanner_probe", {})
    return {
        "encoder": extraction.get("model", path.parent.name),
        "feature_dim": extraction.get("feature_dim"),
        "eval_split": payload.get("eval_split"),
        "n_eval_slides": payload.get("n_eval_slides"),
        "n_eval_regions": payload.get("n_eval_regions"),
        "pair_cosine_average": payload.get("pair_cosine_average"),
        "pair_cosine_worst": payload.get("pair_cosine_worst"),
        "retrieval_top1_average": payload.get("retrieval_top1_average"),
        "retrieval_top1_worst": payload.get("retrieval_top1_worst"),
        "retrieval_mrr_average": payload.get("retrieval_mrr_average"),
        "scanner_probe_accuracy": probe.get("accuracy"),
        "scanner_probe_balanced_accuracy": probe.get("balanced_accuracy"),
        "effective_rank": payload.get("effective_rank"),
        "feature_variance_mean": payload.get("feature_variance_mean"),
        "summary_path": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [load_summary(path) for path in args.summaries]
    frame = pd.DataFrame(rows)
    if frame["eval_split"].nunique(dropna=False) != 1:
        raise SystemExit("Refusing to compare summaries from different evaluation splits.")
    if frame["n_eval_slides"].nunique(dropna=False) != 1:
        raise SystemExit("Refusing to compare summaries with different slide counts.")

    frame = frame.sort_values(
        ["scanner_probe_accuracy", "retrieval_top1_average"],
        ascending=[True, False],
    ).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "frozen_encoder_comparison.csv", index=False)

    best_scanner = frame.iloc[0]
    best_retrieval = frame.sort_values("retrieval_top1_average", ascending=False).iloc[0]
    lines = [
        "# SCORPION frozen encoder comparison",
        "",
        f"Evaluation split: `{frame.iloc[0]['eval_split']}`",
        "",
        frame.drop(columns=["summary_path"]).to_markdown(index=False),
        "",
        "## Descriptive observations",
        "",
        f"- Lowest scanner-probe accuracy: `{best_scanner['encoder']}` ({best_scanner['scanner_probe_accuracy']:.4f}).",
        f"- Highest cross-scanner top-1 retrieval: `{best_retrieval['encoder']}` ({best_retrieval['retrieval_top1_average']:.4f}).",
        "- These are frozen descriptive baselines, not a model-selection result on the untouched test split.",
    ]
    (args.out_dir / "frozen_encoder_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(f"Wrote {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
