#!/usr/bin/env python3
"""Run stronger probe attacks on the canine PathoAlign separation result.

This script tests whether the biological/acquisition separation survives probes
stronger than the default linear logistic audit.

Targets:
- scanner_id with sample_id-blocked CV: can acquisition identity be decoded while
  holding out biological samples?
- sample_id with scanner_id-blocked CV: can biological sample identity be decoded
  across held-out scanners?

Representations:
- raw DINOv2 features;
- paired-reference projected features;
- PathoAlign biological features;
- PathoAlign acquisition features.

Probes:
- logistic linear probe;
- MLP;
- random forest;
- kNN.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DINOV2 = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFEST = Path("results/external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv")
RUN_ROOT = Path("results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stronger probe battery on canine PathoAlign representations.")
    parser.add_argument("--out", type=Path, default=Path("tmp/canine_stronger_probe_battery"))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="911,912,913,914,915")
    parser.add_argument("--probes", default="linear,mlp,random_forest,knn")
    parser.add_argument("--targets", default="scanner_id,sample_id")
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--mlp-max-iter", type=int, default=300)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run fold 0 and seeds 911,912 only.")
    parser.add_argument(
        "--no-random-label-control",
        action="store_true",
        help="Skip random-label refits. Useful for very slow full nonlinear sweeps.",
    )
    return parser.parse_args()


def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def derive_sample_id(region_id: pd.Series) -> pd.Series:
    return region_id.astype(str).str.replace(r"__region_.*$", "", regex=True)


def load_manifest(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    for col in ["sample_id", "region_id", "scanner_id"]:
        if col not in m.columns:
            raise KeyError(f"Required manifest column missing: {col}")
        m[col] = normalize_text(m[col])
    keep = [c for c in ["sample_id", "region_id", "scanner_id", "category_name", "category_id", "fold"] if c in m.columns]
    return m[keep].drop_duplicates(subset=["region_id", "scanner_id"])


def load_npz_table(npz_path: Path, feature_key: str, manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path, allow_pickle=True)
    if feature_key not in z.files:
        raise KeyError(f"Feature key '{feature_key}' not found in {npz_path}. Available keys: {z.files}")
    X = np.asarray(z[feature_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Feature array must be 2D, got {X.shape} from {npz_path}:{feature_key}")
    n = X.shape[0]

    meta: dict[str, Any] = {}
    for key in z.files:
        arr = np.asarray(z[key])
        if arr.shape == (n,):
            meta[key] = arr.astype(str).tolist() if arr.dtype.kind in {"U", "S", "O"} else arr.tolist()
    df = pd.DataFrame(meta)
    for col in ["region_id", "scanner_id", "slide_id", "split", "path", "source_filename"]:
        if col in df.columns:
            df[col] = normalize_text(df[col])
    if "region_id" not in df.columns or "scanner_id" not in df.columns:
        raise KeyError(f"NPZ metadata must include region_id and scanner_id: {npz_path}")
    df = df.merge(manifest, on=["region_id", "scanner_id"], how="left", suffixes=("", "_manifest"))
    if "sample_id" not in df.columns or df["sample_id"].isna().any():
        df["sample_id"] = derive_sample_id(df["region_id"])
    df["sample_id"] = normalize_text(df["sample_id"])
    return X, df


def make_probe(name: str, args: argparse.Namespace, random_state: int):
    if name == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                C=1.0,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(args.mlp_hidden,),
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=args.mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=random_state,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.rf_trees,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=args.knn_neighbors, weights="distance"),
        )
    raise ValueError(f"Unknown probe: {name}")


def target_group_for(target: str) -> str:
    if target == "scanner_id":
        return "sample_id"
    if target == "sample_id":
        return "scanner_id"
    raise ValueError(f"Unknown target: {target}")


def evaluate_probe(
    X: np.ndarray,
    metadata: pd.DataFrame,
    target: str,
    probe_name: str,
    args: argparse.Namespace,
    random_state: int,
) -> dict[str, Any]:
    group_col = target_group_for(target)
    for col in [target, group_col]:
        if col not in metadata.columns:
            raise KeyError(f"Metadata column '{col}' missing.")

    y_raw = normalize_text(metadata[target])
    groups_raw = normalize_text(metadata[group_col])
    valid = y_raw.notna() & groups_raw.notna()
    Xv = X[valid.to_numpy()]
    y_text = y_raw[valid].to_numpy()
    groups = groups_raw[valid].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    n_classes = len(le.classes_)
    n_groups = len(np.unique(groups))
    if n_classes < 2 or n_groups < 2:
        return {
            "status": "skipped",
            "reason": f"Need >=2 classes and groups, got classes={n_classes}, groups={n_groups}",
            "target": target,
            "group_column": group_col,
            "probe": probe_name,
        }

    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    estimator = make_probe(probe_name, args, random_state)
    rng = np.random.default_rng(random_state)
    y_perm = rng.permutation(y)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_perm_true_all: list[int] = []
    y_perm_pred_all: list[int] = []

    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in cv.split(Xv, y, groups):
            clf = clone(estimator)
            clf.fit(Xv[train_idx], y[train_idx])
            pred = clf.predict(Xv[test_idx])
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())

            if not args.no_random_label_control:
                perm_clf = clone(estimator)
                perm_clf.fit(Xv[train_idx], y_perm[train_idx])
                perm_pred = perm_clf.predict(Xv[test_idx])
                y_perm_true_all.extend(y_perm[test_idx].tolist())
                y_perm_pred_all.extend(perm_pred.tolist())

    counts = pd.Series(y_text).value_counts().to_dict()
    majority = max(counts.values()) / len(y_text)
    elapsed = time.time() - start
    result = {
        "status": "ok",
        "target": target,
        "group_column": group_col,
        "probe": probe_name,
        "n_units": int(len(y_text)),
        "n_features": int(Xv.shape[1]),
        "n_classes": int(n_classes),
        "n_groups": int(n_groups),
        "cv_folds": int(n_splits),
        "majority_baseline": float(majority),
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "elapsed_seconds": float(elapsed),
    }
    if args.no_random_label_control:
        result.update({
            "random_label_accuracy": np.nan,
            "random_label_balanced_accuracy": np.nan,
        })
    else:
        result.update({
            "random_label_accuracy": float(accuracy_score(y_perm_true_all, y_perm_pred_all)),
            "random_label_balanced_accuracy": float(balanced_accuracy_score(y_perm_true_all, y_perm_pred_all)),
        })
    return result


def representation_specs(folds: list[int], seeds: list[int]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "raw_dinov2_features",
            "representation": "raw_dinov2_features",
            "fold": None,
            "seed": None,
            "npz": RAW_DINOV2,
            "feature_key": "features",
        }
    ]
    for fold in folds:
        for seed in seeds:
            fold_root = RUN_ROOT / f"fold_{fold}" / "runs"
            specs.append({
                "name": f"fold_{fold}_paired_reference_seed_{seed}",
                "representation": "paired_reference_features",
                "fold": fold,
                "seed": seed,
                "npz": fold_root / f"paired_reference_seed_{seed}" / "projected_features.npz",
                "feature_key": "features",
            })
            specs.append({
                "name": f"fold_{fold}_pathoalign_seed_{seed}_biological_features",
                "representation": "pathoalign_biological_features",
                "fold": fold,
                "seed": seed,
                "npz": fold_root / f"pathoalign_dep20_seed_{seed}" / "projected_features.npz",
                "feature_key": "features",
            })
            specs.append({
                "name": f"fold_{fold}_pathoalign_seed_{seed}_acquisition_features",
                "representation": "pathoalign_acquisition_features",
                "fold": fold,
                "seed": seed,
                "npz": fold_root / f"pathoalign_dep20_seed_{seed}" / "projected_features.npz",
                "feature_key": "acquisition_features",
            })
    return specs


def summarize(per_run: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        per_run[per_run["status"] == "ok"]
        .groupby(group_cols, dropna=False)
        .agg(
            n_runs=("name", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            random_label_accuracy_mean=("random_label_accuracy", "mean"),
            majority_baseline_mean=("majority_baseline", "mean"),
            elapsed_seconds_sum=("elapsed_seconds", "sum"),
        )
        .reset_index()
    )
    order = {
        "raw_dinov2_features": 0,
        "paired_reference_features": 1,
        "pathoalign_biological_features": 2,
        "pathoalign_acquisition_features": 3,
    }
    grouped["_order"] = grouped["representation"].map(order).fillna(99)
    sort_cols = [c for c in ["target", "probe", "_order"] if c in grouped.columns]
    return grouped.sort_values(sort_cols).drop(columns=["_order"])


def fmt(x: Any, digits: int = 6) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def write_markdown(grouped: pd.DataFrame, out: Path) -> None:
    lines: list[str] = []
    lines.append("# Canine stronger probe battery")
    lines.append("")
    lines.append("This battery attacks the PathoAlign separation result with stronger probes than the default linear audit.")
    lines.append("")
    lines.append("Targets:")
    lines.append("")
    lines.append("- `scanner_id`, evaluated with `sample_id`-blocked cross-validation.")
    lines.append("- `sample_id`, evaluated with `scanner_id`-blocked cross-validation.")
    lines.append("")
    lines.append("## Grouped probe results")
    lines.append("")
    lines.append("| Target | Probe | Representation | Runs | Accuracy | Balanced accuracy | Random-label accuracy | Majority baseline |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for _, r in grouped.iterrows():
        lines.append(
            "| {target} | {probe} | {rep} | {n} | {acc} | {bal} | {rand} | {base} |".format(
                target=r["target"],
                probe=r["probe"],
                rep=r["representation"],
                n=int(r["n_runs"]),
                acc=fmt(r["accuracy_mean"]),
                bal=fmt(r["balanced_accuracy_mean"]),
                rand=fmt(r["random_label_accuracy_mean"]),
                base=fmt(r["majority_baseline_mean"]),
            )
        )
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append("This is a probe-based representation-identifiability stress test, not clinical validation.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folds = [int(s.strip()) for s in args.folds.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.quick:
        folds = [0]
        seeds = [911, 912]
    probes = [s.strip() for s in args.probes.split(",") if s.strip()]
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]

    per_run_path = args.out / "canine_stronger_probe_battery_per_run.csv"
    if args.skip_existing and per_run_path.exists():
        per_run = pd.read_csv(per_run_path)
    else:
        manifest = load_manifest(MANIFEST)
        rows: list[dict[str, Any]] = []
        specs = representation_specs(folds, seeds)
        total = len(specs) * len(targets) * len(probes)
        idx = 0
        cache: dict[tuple[str, str], tuple[np.ndarray, pd.DataFrame]] = {}
        for spec in specs:
            if not spec["npz"].exists():
                if args.allow_missing:
                    print(f"[probe-battery] missing, skipping: {spec['npz']}")
                    continue
                raise FileNotFoundError(spec["npz"])
            cache_key = (str(spec["npz"]), spec["feature_key"])
            if cache_key not in cache:
                print(f"[probe-battery] loading {spec['npz']} key={spec['feature_key']}")
                cache[cache_key] = load_npz_table(spec["npz"], spec["feature_key"], manifest)
            X, metadata = cache[cache_key]
            for target in targets:
                for probe in probes:
                    idx += 1
                    print(f"[probe-battery] {idx}/{total} {spec['name']} target={target} probe={probe}")
                    try:
                        result = evaluate_probe(
                            X=X,
                            metadata=metadata,
                            target=target,
                            probe_name=probe,
                            args=args,
                            random_state=args.seed + idx,
                        )
                    except Exception as exc:
                        if not args.allow_missing:
                            raise
                        result = {"status": "error", "reason": repr(exc), "target": target, "probe": probe}
                    row = {
                        "name": spec["name"],
                        "representation": spec["representation"],
                        "fold": spec["fold"],
                        "seed": spec["seed"],
                        "npz": str(spec["npz"]),
                        "feature_key": spec["feature_key"],
                    }
                    row.update(result)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(per_run_path, index=False)
        per_run = pd.DataFrame(rows)
        per_run.to_csv(per_run_path, index=False)

    grouped = summarize(per_run, ["target", "probe", "representation"])
    grouped_path = args.out / "canine_stronger_probe_battery_grouped.csv"
    grouped.to_csv(grouped_path, index=False)

    target_grouped = summarize(per_run, ["target", "representation"])
    target_grouped_path = args.out / "canine_stronger_probe_battery_by_target.csv"
    target_grouped.to_csv(target_grouped_path, index=False)

    md_path = args.out / "canine_stronger_probe_battery_summary.md"
    write_markdown(grouped, md_path)

    print(f"[probe-battery] wrote {per_run_path}")
    print(f"[probe-battery] wrote {grouped_path}")
    print(f"[probe-battery] wrote {target_grouped_path}")
    print(f"[probe-battery] wrote {md_path}")
    print("\nTARGET SUMMARY")
    print(target_grouped.to_string(index=False))
    print("\nGROUP SUMMARY")
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
