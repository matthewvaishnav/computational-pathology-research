"""
Run a simple CAMELYON slide-level baseline from cached patch features.

This script trains a lightweight slide-level classifier using pre-extracted
HDF5 patch features plus simple slide aggregation (mean or max). It also
supports exporting per-tile scores from the trained linear model so the
existing CAMELYON heatmap pipeline can visualize model-driven outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# Add repo root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data.render_camelyon_heatmap import build_camelyon_heatmap_artifacts
from src.data.camelyon_dataset import CAMELYONPatchDataset, CAMELYONSlideIndex

AggregationMethod = Literal["mean", "max"]


def collect_slide_level_features(
    slide_index_path: Union[str, Path],
    features_dir: Union[str, Path],
    split: str,
    aggregation: AggregationMethod = "mean",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load aggregated slide features, labels, and slide IDs for one split."""
    slide_index = CAMELYONSlideIndex.load(slide_index_path)
    dataset = CAMELYONPatchDataset(
        slide_index=slide_index,
        features_dir=features_dir,
        split=split,
    )

    slide_features: List[np.ndarray] = []
    labels: List[int] = []
    slide_ids: List[str] = []

    for slide in dataset.slides:
        aggregated = dataset.aggregate_slide_features(slide.slide_id, method=aggregation)
        if aggregated is None:
            continue
        if slide.label < 0:
            continue

        slide_features.append(aggregated.detach().cpu().numpy())
        labels.append(int(slide.label))
        slide_ids.append(slide.slide_id)

    if not slide_features:
        raise ValueError(f"No usable labeled CAMELYON slides found for split '{split}'.")

    return (
        np.stack(slide_features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        slide_ids,
    )


def train_logistic_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a slide-level logistic baseline."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError(
            "Training split must contain at least two classes for logistic regression."
        )

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(features, labels)
    return model


def evaluate_slide_classifier(
    model: LogisticRegression,
    features: np.ndarray,
    labels: np.ndarray,
    slide_ids: List[str],
) -> Dict[str, Any]:
    """Evaluate a binary slide-level classifier."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    if np.unique(labels).size < 2:
        auc = None
    else:
        auc = float(roc_auc_score(labels, probabilities))

    return {
        "num_slides": int(len(labels)),
        "slide_ids": slide_ids,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "labels": labels.tolist(),
    }


def save_slide_predictions_csv(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save per-slide evaluation predictions to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slide_id", "label", "prediction", "probability"],
        )
        writer.writeheader()
        for slide_id, label, prediction, probability in zip(
            metrics["slide_ids"],
            metrics["labels"],
            metrics["predictions"],
            metrics["probabilities"],
        ):
            writer.writerow(
                {
                    "slide_id": slide_id,
                    "label": int(label),
                    "prediction": int(prediction),
                    "probability": float(probability),
                }
            )


def plot_confusion_matrix(cm: np.ndarray, output_path: Union[str, Path]) -> bool:
    """Render a binary confusion matrix plot if plotting dependencies exist."""
    if not PLOT_AVAILABLE:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        cbar_kws={"label": "Count"},
    )
    plt.title("CAMELYON Feature Baseline Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_roc_curve_for_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> bool:
    """Render an ROC curve for evaluation metrics when AUC is defined."""
    if not PLOT_AVAILABLE or metrics["auc"] is None:
        return False

    labels = np.asarray(metrics["labels"], dtype=np.int64)
    if np.unique(labels).size < 2:
        return False

    probabilities = np.asarray(metrics["probabilities"], dtype=np.float32)
    fpr, tpr, _ = roc_curve(labels, probabilities)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CAMELYON Feature Baseline ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def export_model_tile_scores(
    feature_file: Union[str, Path],
    model: LogisticRegression,
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Export per-tile scores from a trained linear slide baseline."""
    feature_file = Path(feature_file)
    output_path = Path(output_path)

    from scripts.data.export_camelyon_tile_scores import compute_tile_scores  # local reuse guard
    import h5py

    with h5py.File(feature_file, "r") as handle:
        features = handle["features"][:].astype(np.float32)
        coordinates = handle["coordinates"][:].astype(np.float32)

    logits = features @ model.coef_[0].astype(np.float32) + float(model.intercept_[0])
    scores = 1.0 / (1.0 + np.exp(-logits))
    # Reuse utility import to keep the same validation expectations around arrays.
    _ = compute_tile_scores(features, method="mean_activation")

    tiles = [
        {"x": float(coord[0]), "y": float(coord[1]), "score": float(score)}
        for coord, score in zip(coordinates, scores)
    ]
    payload = {
        "source_feature_file": feature_file.as_posix(),
        "score_source": "logistic_regression",
        "num_tiles": int(len(tiles)),
        "tiles": tiles,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def run_camelyon_feature_baseline(
    slide_index_path: Union[str, Path],
    features_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    train_split: str = "train",
    eval_split: str = "val",
    aggregation: AggregationMethod = "mean",
    export_tile_scores_for_slide: Union[str, None] = None,
    heatmap_slide_width: Union[int, None] = None,
    heatmap_slide_height: Union[int, None] = None,
    heatmap_patch_size: Union[int, None] = None,
    heatmap_thumbnail_path: Union[Union[str, Path], None] = None,
    heatmap_annotation_xml_path: Union[Union[str, Path], None] = None,
    heatmap_downsample: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate the CAMELYON feature baseline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_slide_ids = collect_slide_level_features(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        split=train_split,
        aggregation=aggregation,
    )
    eval_x, eval_y, eval_slide_ids = collect_slide_level_features(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        split=eval_split,
        aggregation=aggregation,
    )

    model = train_logistic_baseline(train_x, train_y, random_state=random_state)
    train_metrics = evaluate_slide_classifier(model, train_x, train_y, train_slide_ids)
    eval_metrics = evaluate_slide_classifier(model, eval_x, eval_y, eval_slide_ids)

    model_path = output_dir / "camelyon_feature_baseline.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    predictions_csv_path = output_dir / "eval_slide_predictions.csv"
    save_slide_predictions_csv(eval_metrics, predictions_csv_path)

    confusion_matrix_path = output_dir / "confusion_matrix.png"
    roc_curve_path = output_dir / "roc_curve.png"
    confusion_matrix_generated = plot_confusion_matrix(
        np.asarray(eval_metrics["confusion_matrix"], dtype=np.int64),
        confusion_matrix_path,
    )
    roc_curve_generated = plot_roc_curve_for_metrics(eval_metrics, roc_curve_path)

    results = {
        "train_split": train_split,
        "eval_split": eval_split,
        "aggregation": aggregation,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "artifacts": {
            "model": model_path.as_posix(),
            "eval_slide_predictions": predictions_csv_path.as_posix(),
        },
    }
    if confusion_matrix_generated:
        results["artifacts"]["confusion_matrix"] = confusion_matrix_path.as_posix()
    if roc_curve_generated:
        results["artifacts"]["roc_curve"] = roc_curve_path.as_posix()

    if export_tile_scores_for_slide is not None:
        slide_feature_file = Path(features_dir) / f"{export_tile_scores_for_slide}.h5"
        tile_scores_path = output_dir / f"{export_tile_scores_for_slide}_tile_scores.json"
        export_model_tile_scores(
            feature_file=slide_feature_file,
            model=model,
            output_path=tile_scores_path,
        )
        results["artifacts"]["tile_scores"] = tile_scores_path.as_posix()

        if heatmap_slide_width is not None and heatmap_slide_height is not None:
            if heatmap_patch_size is None:
                raise ValueError(
                    "heatmap_patch_size must be provided when rendering model-driven heatmaps."
                )
            heatmap_output_dir = output_dir / "heatmaps" / export_tile_scores_for_slide
            heatmap_summary = build_camelyon_heatmap_artifacts(
                tile_scores_path=tile_scores_path,
                slide_width=heatmap_slide_width,
                slide_height=heatmap_slide_height,
                patch_size=heatmap_patch_size,
                output_dir=heatmap_output_dir,
                thumbnail_path=heatmap_thumbnail_path,
                annotation_xml_path=heatmap_annotation_xml_path,
                downsample=heatmap_downsample,
            )
            results["artifacts"]["heatmap_summary"] = heatmap_summary["summary_path"]

    results_path = output_dir / "camelyon_feature_baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    results["artifacts"]["results"] = results_path.as_posix()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CAMELYON slide-level baseline from cached patch features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/run_camelyon_feature_baseline.py ^
      --slide-index data/camelyon/slide_index.json ^
      --features-dir data/camelyon/features ^
      --output-dir results/camelyon/baseline

  python experiments/run_camelyon_feature_baseline.py ^
      --slide-index data/camelyon/slide_index.json ^
      --features-dir data/camelyon/features ^
      --output-dir results/camelyon/baseline ^
      --aggregation max --export-tile-scores-for-slide tumor_001

  python experiments/run_camelyon_feature_baseline.py ^
      --slide-index data/camelyon/slide_index.json ^
      --features-dir data/camelyon/features ^
      --output-dir results/camelyon/baseline ^
      --export-tile-scores-for-slide tumor_001 ^
      --heatmap-slide-width 80000 --heatmap-slide-height 60000 ^
      --heatmap-patch-size 256 --heatmap-downsample 32
        """,
    )
    parser.add_argument("--slide-index", required=True, help="Path to CAMELYON slide index JSON")
    parser.add_argument(
        "--features-dir", required=True, help="Directory of slide HDF5 feature files"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for model/results artifacts")
    parser.add_argument("--train-split", default="train", help="Training split name")
    parser.add_argument("--eval-split", default="val", help="Evaluation split name")
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max"],
        default="mean",
        help="Slide-level patch aggregation method",
    )
    parser.add_argument(
        "--export-tile-scores-for-slide",
        default=None,
        help="Optional slide ID for exporting model-driven tile scores",
    )
    parser.add_argument(
        "--heatmap-slide-width",
        type=int,
        default=None,
        help="Optional slide width for heatmap rendering",
    )
    parser.add_argument(
        "--heatmap-slide-height",
        type=int,
        default=None,
        help="Optional slide height for heatmap rendering",
    )
    parser.add_argument(
        "--heatmap-patch-size", type=int, default=None, help="Patch size for heatmap rendering"
    )
    parser.add_argument(
        "--heatmap-thumbnail", default=None, help="Optional thumbnail image for heatmap overlay"
    )
    parser.add_argument(
        "--heatmap-annotation-xml",
        default=None,
        help="Optional CAMELYON XML annotation for heatmap scoring",
    )
    parser.add_argument(
        "--heatmap-downsample", type=int, default=1, help="Downsample factor for rendered heatmaps"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = run_camelyon_feature_baseline(
        slide_index_path=args.slide_index,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        aggregation=args.aggregation,
        export_tile_scores_for_slide=args.export_tile_scores_for_slide,
        heatmap_slide_width=args.heatmap_slide_width,
        heatmap_slide_height=args.heatmap_slide_height,
        heatmap_patch_size=args.heatmap_patch_size,
        heatmap_thumbnail_path=args.heatmap_thumbnail,
        heatmap_annotation_xml_path=args.heatmap_annotation_xml,
        heatmap_downsample=args.heatmap_downsample,
        random_state=args.random_state,
    )
    print(f"Saved CAMELYON baseline results to {results['artifacts']['results']}")


if __name__ == "__main__":
    main()
