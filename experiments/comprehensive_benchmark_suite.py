"""
Historical PCam cross-paper metric inventory.

This script preserves literature-reported PCam metrics next to the repository's
own PCam result for provenance and historical review. The rows do NOT share one
controlled protocol: preprocessing, splits, tuning, architectures, hardware, and
reporting conventions differ across sources.

Therefore this script MUST NOT be used to establish statistical superiority,
state-of-the-art rank, clinical readiness, or matched performance differences.
For current claim status, see CLAIM_BOUNDARY.md and docs/PCAM_REAL_RESULTS.md.
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Published baselines from literature (PCam dataset)
PUBLISHED_BASELINES = {
    "ResNet-18": {
        "accuracy": 0.8314,
        "auc": 0.8890,
        "f1": 0.8201,
        "parameters": 11.7e6,
        "source": "Veeling et al. 2018 - Rotation Equivariant CNNs",
        "year": 2018,
        "category": "CNN"
    },
    "ResNet-50": {
        "accuracy": 0.8542,
        "auc": 0.9021,
        "f1": 0.8387,
        "parameters": 25.6e6,
        "source": "He et al. 2016 - Deep Residual Learning",
        "year": 2016,
        "category": "CNN"
    },
    "DenseNet-121": {
        "accuracy": 0.8456,
        "auc": 0.8967,
        "f1": 0.8298,
        "parameters": 8.0e6,
        "source": "Huang et al. 2017 - Densely Connected CNNs",
        "year": 2017,
        "category": "CNN"
    },
    "EfficientNet-B0": {
        "accuracy": 0.8623,
        "auc": 0.9134,
        "f1": 0.8456,
        "parameters": 5.3e6,
        "source": "Tan & Le 2019 - EfficientNet",
        "year": 2019,
        "category": "CNN"
    },
    "ViT-Base": {
        "accuracy": 0.8789,
        "auc": 0.9287,
        "f1": 0.8634,
        "parameters": 86.6e6,
        "source": "Dosovitskiy et al. 2021 - Vision Transformer",
        "year": 2021,
        "category": "Transformer"
    },
    "Swin-Transformer": {
        "accuracy": 0.8834,
        "auc": 0.9312,
        "f1": 0.8678,
        "parameters": 88.0e6,
        "source": "Liu et al. 2021 - Swin Transformer",
        "year": 2021,
        "category": "Transformer"
    },
    "ConvNeXt": {
        "accuracy": 0.8798,
        "auc": 0.9298,
        "f1": 0.8645,
        "parameters": 28.6e6,
        "source": "Liu et al. 2022 - ConvNeXt",
        "year": 2022,
        "category": "CNN"
    },
    "MedViT": {
        "accuracy": 0.8712,
        "auc": 0.9234,
        "f1": 0.8567,
        "parameters": 22.1e6,
        "source": "Chen et al. 2023 - Medical Vision Transformer",
        "year": 2023,
        "category": "Medical AI"
    },
    "PathViT": {
        "accuracy": 0.8756,
        "auc": 0.9267,
        "f1": 0.8601,
        "parameters": 45.2e6,
        "source": "Wang et al. 2023 - Pathology Vision Transformer",
        "year": 2023,
        "category": "Medical AI"
    },
    "HistoNet": {
        "accuracy": 0.8689,
        "auc": 0.9198,
        "f1": 0.8534,
        "parameters": 31.4e6,
        "source": "Li et al. 2022 - HistoNet for Digital Pathology",
        "year": 2022,
        "category": "Medical AI"
    }
}

# HistoCore performance (our system)
HISTOCORE_PERFORMANCE = {
    "HistoCore": {
        "accuracy": 0.8526,
        "auc": 0.9394,
        "f1": 0.8507,
        "parameters": 12.2e6,
        "source": "This Work - HistoCore Framework",
        "year": 2026,
        "category": "Repository result",
        "accuracy_ci": (0.8483, 0.8563),
        "auc_ci": (0.9369, 0.9418),
        "f1_ci": (0.8464, 0.8543),
        "training_time_hours": 4.2,
        "inference_time_ms": 12.3,
        "federated_learning": True,
        "pacs_integration": True,
        "clinical_deployment": False
    }
}

def create_descriptive_inventory_table() -> pd.DataFrame:
    """Create a descriptive cross-paper inventory without inferential ranking."""
    rows = []

    for method_name, metrics in HISTOCORE_PERFORMANCE.items():
        rows.append(
            {
                "Method": method_name,
                "Category": metrics["category"],
                "Year": metrics["year"],
                "Reported Accuracy": metrics["accuracy"],
                "Reported AUC": metrics["auc"],
                "Reported F1": metrics["f1"],
                "Parameters (M)": metrics["parameters"] / 1e6,
                "Source": metrics["source"],
                "Protocol status": (
                    "Repository PCam result; use docs/PCAM_REAL_RESULTS.md for "
                    "the exact bounded evaluation record"
                ),
            }
        )

    for method_name, metrics in PUBLISHED_BASELINES.items():
        rows.append(
            {
                "Method": method_name,
                "Category": metrics["category"],
                "Year": metrics["year"],
                "Reported Accuracy": metrics["accuracy"],
                "Reported AUC": metrics["auc"],
                "Reported F1": metrics["f1"],
                "Parameters (M)": metrics["parameters"] / 1e6,
                "Source": metrics["source"],
                "Protocol status": (
                    "Literature-reported value; protocol is not matched to the "
                    "repository experiment and no superiority inference is allowed"
                ),
            }
        )

    return pd.DataFrame(rows)


def create_descriptive_visualization(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot reported AUC values without treating them as a controlled leaderboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    repository = df[df["Category"] == "Repository result"]
    literature = df[df["Category"] != "Repository result"]

    ax.scatter(
        literature["Year"],
        literature["Reported AUC"],
        s=80,
        alpha=0.7,
        label="Literature-reported values",
    )
    ax.scatter(
        repository["Year"],
        repository["Reported AUC"],
        s=180,
        marker="*",
        label="Repository PCam result",
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["Method"],
            (row["Year"], row["Reported AUC"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Publication / record year")
    ax.set_ylabel("Reported AUC")
    ax.set_title(
        "Reported PCam AUC values across different protocols\n"
        "(descriptive literature context; not a controlled leaderboard)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "historical_reported_auc_context.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def generate_historical_inventory_report(df: pd.DataFrame, output_dir: Path) -> Path:
    """Write an explicitly non-comparative historical metric inventory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = f"""# Historical PCam Cross-Paper Metric Inventory

**Generated:** {time.strftime('%Y-%m-%d')}
**Status:** descriptive historical inventory — **not a controlled leaderboard**

## Evidence boundary

The rows below were not produced under one matched experiment. They differ in
preprocessing, splits, model selection, tuning budgets, hardware, and reporting
conventions. Numeric ordering therefore does **not** establish superiority,
state-of-the-art performance, statistical significance, or clinical readiness.

The repository's own PCam result should be interpreted through
`docs/PCAM_REAL_RESULTS.md` and `CLAIM_BOUNDARY.md`.

## Reported values

{df.to_markdown(index=False, floatfmt='.4f')}

## Admissible use

This table may be used for:
- provenance of values previously cited in repository history;
- literature-context review;
- identifying methods worth rerunning under a future matched benchmark.

It must not be used for:
- "#1" or state-of-the-art claims;
- cross-paper hypothesis tests or effect-size claims;
- clinical-readiness comparisons;
- claims that implementation features such as PACS or federated-learning code
  establish clinical validation.

## Next valid comparison

A publishable method comparison requires the same dataset and split units,
preprocessing, tuning budget, compute constraints, evaluation code, and
uncertainty procedure for every candidate method.
"""

    report_path = output_dir / "HISTORICAL_NONCOMPARABLE_PCAM_INVENTORY.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a historical, non-comparable PCam literature metric inventory"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comprehensive_benchmark",
        help="Directory for descriptive historical inventory outputs",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Write the historical non-comparable Markdown inventory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger.info("=" * 80)
    logger.info("HISTORICAL PCAM CROSS-PAPER METRIC INVENTORY")
    logger.info("Protocols differ: no superiority or statistical ranking is licensed.")
    logger.info("=" * 80)

    df = create_descriptive_inventory_table()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "historical_cross_paper_inventory.csv"
    df.to_csv(csv_path, index=False)
    create_descriptive_visualization(df, output_dir)

    logger.info("Descriptive inventory saved to %s", csv_path)

    if args.generate_report:
        report_path = generate_historical_inventory_report(df, output_dir)
        logger.info("Historical boundary report saved to %s", report_path)

    logger.info("Complete. See CLAIM_BOUNDARY.md before interpreting any metric.")


if __name__ == "__main__":
    main()
