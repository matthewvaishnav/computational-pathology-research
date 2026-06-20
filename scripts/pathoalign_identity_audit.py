#!/usr/bin/env python3
"""PathoAlign Identity Audit v0.

Audits frozen pathology representations for shortcut identity and biological
preservation. This is the first runnable companion to the PathoAlign Oncology
Identity Benchmark.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # older sklearn
    StratifiedGroupKFold = None  # type: ignore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID_CANDIDATES = ["unit_id", "feature_id", "patch_id", "region_id", "sample_id", "slide_id", "id"]
SHORTCUT_CANDIDATES = [
    "scanner_id", "scanner", "site_id", "site", "hospital_id", "hospital", "stain_id",
    "stain", "client_id", "client", "lab_id", "lab", "cohort_id", "cohort",
    "dataset_id", "dataset", "annotation_source", "annotator_id",
]
BIOLOGY_CANDIDATES = [
    "region_id", "sample_id", "patient_id", "case_id", "tissue_id", "biology_label",
    "disease_label", "tumor_label", "task_label", "slide_label",
]


def csv_list(value: str | None) -> list[str] | None:
    return None if value is None or value.strip() == "" else [v.strip() for v in value.split(",") if v.strip()]


def pick_id(df: pd.DataFrame, requested: str | None, name: str) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {name} id column '{requested}' was not found.")
        return requested
    return next((c for c in ID_CANDIDATES if c in df.columns), None)


def present(df: pd.DataFrame, requested: list[str] | None, candidates: list[str]) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested columns were not found: {missing}")
        return requested
    return [c for c in candidates if c in df.columns]


def load_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, list[str], str]:
    f = pd.read_csv(args.features)
    m = pd.read_csv(args.metadata)
    fid = pick_id(f, args.feature_id_column, "feature")
    mid = pick_id(m, args.metadata_id_column, "metadata")

    if fid and mid:
        if f[fid].duplicated().any() or m[mid].duplicated().any():
            raise ValueError("Identifier columns must be unique in both input tables.")
        df = m.merge(f, left_on=mid, right_on=fid, how="inner", suffixes=("", "_feature"))
        if len(df) == 0:
            raise ValueError("Feature and metadata tables had no matching identifiers.")
        audit_id = mid
        if fid != mid and fid in df.columns:
            df = df.drop(columns=[fid])
    else:
        if len(f) != len(m):
            raise ValueError("No shared id column found and feature/metadata row counts differ.")
        audit_id = "audit_row_id"
        f = f.copy(); m = m.copy()
        f[audit_id] = np.arange(len(f)); m[audit_id] = np.arange(len(m))
        df = m.merge(f, on=audit_id, how="inner")

    requested_features = csv_list(args.feature_columns)
    if requested_features:
        missing = [c for c in requested_features if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns were not found: {missing}")
        feature_cols = requested_features
    else:
        meta_cols = set(m.columns)
        feature_cols = [c for c in f.columns if c != audit_id and c not in meta_cols and pd.api.types.is_numeric_dtype(f[c])]
    if not feature_cols:
        raise ValueError("No numeric feature columns found. Pass --feature-columns f0,f1,...")

    X = df[feature_cols].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        print(f"[audit] dropping {int((~ok).sum())} rows with NaN/Inf features", file=sys.stderr)
        df = df.loc[ok].reset_index(drop=True); X = X[ok]
    if len(df) < 3:
        raise ValueError("Need at least three valid rows.")
    return df, X, feature_cols, audit_id


def block_column(df: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested block column '{requested}' was not found.")
        return requested
    return next((c for c in ["sample_id", "patient_id", "case_id", "client_id", "site_id"] if c in df.columns), None)


def majority(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def cv_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: int):
    _, counts = np.unique(y, return_counts=True)
    n = min(n_splits, int(counts.min()))
    if n < 2:
        return []
    if groups is not None and len(np.unique(groups)) >= n:
        per_class_groups = [len(np.unique(groups[y == cls])) for cls in np.unique(y)]
        g = min(n, min(per_class_groups))
        if g >= 2:
            if StratifiedGroupKFold is not None:
                return list(StratifiedGroupKFold(n_splits=g, shuffle=True, random_state=seed).split(X, y, groups))
            return list(GroupKFold(n_splits=g).split(X, y, groups))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(X, y))


def boot_ci(values: np.ndarray, groups: np.ndarray | None, n_boot: int, seed: int) -> tuple[float, float]:
    if n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed); scores = []
    if groups is None:
        for _ in range(n_boot):
            scores.append(float(values[rng.integers(0, len(values), len(values))].mean()))
    else:
        unique = np.unique(groups)
        if len(unique) < 2:
            return float("nan"), float("nan")
        idx = {g: np.flatnonzero(groups == g) for g in unique}
        for _ in range(n_boot):
            sampled = rng.choice(unique, len(unique), replace=True)
            rows = np.concatenate([idx[g] for g in sampled])
            scores.append(float(values[rows].mean()))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def run_probe(df: pd.DataFrame, X: np.ndarray, col: str, block: str | None, args: argparse.Namespace):
    mask = df[col].notna().to_numpy(); y = df.loc[mask, col].astype(str).to_numpy(); Xc = X[mask]
    if len(np.unique(y)) < 2:
        return {"column": col, "status": "skipped", "reason": "fewer than two classes", "n_units": int(len(y))}, None
    groups = df.loc[mask, block].astype(str).to_numpy() if block and block in df.columns else None
    splits = cv_splits(Xc, y, groups, args.n_splits, args.seed)
    if not splits:
        return {"column": col, "status": "skipped", "reason": "not enough class/group support", "n_units": int(len(y))}, None

    pred = np.array([None] * len(y), dtype=object)
    rand_pred = np.array([None] * len(y), dtype=object)
    yr = np.random.default_rng(args.seed).permutation(y)
    folds = rand_folds = 0
    for k, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=args.seed + k))
        model.fit(Xc[tr], y[tr]); pred[te] = model.predict(Xc[te]); folds += 1
        if len(np.unique(yr[tr])) >= 2:
            rm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=args.seed + 10000 + k))
            rm.fit(Xc[tr], yr[tr]); rand_pred[te] = rm.predict(Xc[te]); rand_folds += 1

    valid = pred != None  # noqa: E711
    if valid.sum() == 0:
        return {"column": col, "status": "skipped", "reason": "all folds invalid"}, None
    yt = y[valid]; yp = pred[valid].astype(str); gp = groups[valid] if groups is not None else None
    hits = (yt == yp).astype(float); lo, hi = boot_ci(hits, gp, args.n_bootstrap, args.seed + 1)
    rv = rand_pred != None  # noqa: E711
    rand_acc = float(accuracy_score(y[rv], rand_pred[rv].astype(str))) if rv.sum() else float("nan")
    rand_bal = float(balanced_accuracy_score(y[rv], rand_pred[rv].astype(str))) if rv.sum() else float("nan")
    values, counts = np.unique(y, return_counts=True)
    result = {
        "column": col, "status": "ok", "n_units": int(len(yt)), "n_classes": int(len(values)),
        "class_counts": {str(v): int(c) for v, c in zip(values, counts)},
        "block_column": block if gp is not None else None, "n_blocks": int(len(np.unique(gp))) if gp is not None else None,
        "cv_folds": int(folds), "accuracy": float(accuracy_score(yt, yp)), "accuracy_ci_low": lo,
        "accuracy_ci_high": hi, "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "majority_baseline": majority(yt), "random_label_accuracy": rand_acc,
        "random_label_balanced_accuracy": rand_bal, "random_label_cv_folds": int(rand_folds),
    }
    preds = pd.DataFrame({"row_index": df.loc[mask].index.to_numpy()[valid], "probe_column": col, "true_label": yt, "predicted_label": yp})
    if gp is not None: preds["block"] = gp
    return result, preds


def normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n


def retrieval(df: pd.DataFrame, X: np.ndarray, col: str, block: str | None, args: argparse.Namespace) -> dict[str, Any]:
    mask = df[col].notna().to_numpy(); y = df.loc[mask, col].astype(str).to_numpy(); Xc = X[mask]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {"column": col, "status": "skipped", "reason": "not enough labels", "n_units": int(len(y))}
    counts = pd.Series(y).value_counts(); keep = np.array([counts[v] >= 2 for v in y])
    if keep.sum() < 3:
        return {"column": col, "status": "skipped", "reason": "no repeated labels", "n_units": int(len(y))}
    y = y[keep]; Z = normalize(Xc[keep]); sim = Z @ Z.T; np.fill_diagonal(sim, -np.inf)
    hits = (y[np.argmax(sim, axis=1)] == y).astype(float)
    groups = None
    if block and block in df.columns:
        groups = df.iloc[np.flatnonzero(mask)[keep]][block].astype(str).to_numpy()
    lo, hi = boot_ci(hits, groups, args.n_bootstrap, args.seed + 101)
    _, label_counts = np.unique(y, return_counts=True)
    return {"column": col, "status": "ok", "n_units": int(len(y)), "n_classes": int(len(label_counts)),
            "top1_same_label_retrieval": float(hits.mean()), "top1_ci_low": lo, "top1_ci_high": hi,
            "majority_label_baseline": float(label_counts.max() / label_counts.sum()),
            "block_column": block if groups is not None else None, "n_blocks": int(len(np.unique(groups))) if groups is not None else None}


def consistency(df: pd.DataFrame, X: np.ndarray, bio_cols: list[str], shortcut_cols: list[str]) -> list[dict[str, Any]]:
    Z = normalize(X); rows = []
    for b in bio_cols:
        for s in shortcut_cols:
            valid = df[b].notna() & df[s].notna()
            if valid.sum() < 3: continue
            idx = np.flatnonzero(valid.to_numpy()); bv = df.loc[valid, b].astype(str).to_numpy(); sv = df.loc[valid, s].astype(str).to_numpy()
            cos = []; ng = 0
            for val in np.unique(bv):
                local = np.flatnonzero(bv == val)
                if len(local) < 2 or len(np.unique(sv[local])) < 2: continue
                ng += 1
                for i in range(len(local)):
                    for j in range(i + 1, len(local)):
                        if sv[local[i]] != sv[local[j]]:
                            cos.append(float(Z[idx[local[i]]] @ Z[idx[local[j]]]))
            if cos:
                rows.append({"biology_column": b, "shortcut_column": s, "status": "ok",
                             "n_biology_groups": int(ng), "n_cross_acquisition_pairs": int(len(cos)),
                             "mean_cross_acquisition_cosine": float(np.mean(cos)),
                             "median_cross_acquisition_cosine": float(np.median(cos))})
    return rows


def collapse(X: np.ndarray) -> dict[str, Any]:
    C = X - X.mean(axis=0, keepdims=True); var = C.var(axis=0); sv = np.linalg.svd(C, full_matrices=False, compute_uv=False)
    if sv.size and sv.sum() > 0:
        p = sv[sv > 0] / sv[sv > 0].sum(); erank = float(math.exp(-np.sum(p * np.log(p + 1e-12))))
        top = float(sv[0] / sv.sum()); nrank = int((sv > 1e-8).sum())
    else:
        erank = 0.0; top = float("nan"); nrank = 0
    norms = np.linalg.norm(X, axis=1)
    return {"n_units": int(X.shape[0]), "n_features": int(X.shape[1]), "total_variance": float(var.sum()),
            "mean_dimension_variance": float(var.mean()), "zero_variance_dimension_fraction": float((var <= 1e-12).mean()),
            "effective_rank": erank, "numerical_rank": nrank, "top_singular_value_fraction": top,
            "l2_norm_mean": float(norms.mean()), "l2_norm_std": float(norms.std())}


def flatten(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in summary["collapse"].items(): rows.append({"category": "collapse", "measurement": k, "column": "", "value": v, "status": "ok"})
    for cat, items in [("shortcut_probe", summary["shortcut_probes"]), ("biology_retrieval", summary["biology_retrieval"]), ("cross_acquisition_consistency", summary["cross_acquisition_consistency"])]:
        for item in items:
            col = item.get("column") or f"{item.get('biology_column')}__by__{item.get('shortcut_column')}"
            for k, v in item.items():
                if k in {"column", "status", "reason", "class_counts", "biology_column", "shortcut_column"}: continue
                rows.append({"category": cat, "measurement": k, "column": col, "value": v, "status": item.get("status", "")})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Audit pathology representations for shortcut identity and biology preservation.")
    p.add_argument("--features", required=True, type=Path); p.add_argument("--metadata", required=True, type=Path); p.add_argument("--out", required=True, type=Path)
    p.add_argument("--feature-id-column"); p.add_argument("--metadata-id-column"); p.add_argument("--feature-columns")
    p.add_argument("--shortcut-columns"); p.add_argument("--biology-columns"); p.add_argument("--block-column")
    p.add_argument("--n-splits", type=int, default=5); p.add_argument("--n-bootstrap", type=int, default=500); p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    df, X, fcols, audit_id = load_tables(args)
    shortcuts = present(df, csv_list(args.shortcut_columns), SHORTCUT_CANDIDATES)
    biology = present(df, csv_list(args.biology_columns), BIOLOGY_CANDIDATES)
    block = block_column(df, args.block_column)
    summary = {"audit_version": "PathoAlign Identity Audit v0", "inputs": {"features": str(args.features), "metadata": str(args.metadata), "n_units": int(X.shape[0]), "n_features": int(X.shape[1]), "feature_columns": fcols, "audit_id_column": audit_id, "shortcut_columns": shortcuts, "biology_columns": biology, "block_column": block}, "validity_rule": "Shortcut identity must decrease while biology remains recoverable and collapse controls pass.", "collapse": collapse(X), "shortcut_probes": [], "biology_retrieval": [], "cross_acquisition_consistency": []}
    pred_frames = []
    for col in shortcuts:
        res, pred = run_probe(df, X, col, block, args); summary["shortcut_probes"].append(res)
        if pred is not None: pred_frames.append(pred)
    for col in biology:
        summary["biology_retrieval"].append(retrieval(df, X, col, block, args))
    summary["cross_acquisition_consistency"] = consistency(df, X, biology, shortcuts)
    (args.out / "identity_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flatten(summary).to_csv(args.out / "identity_audit_summary.csv", index=False)
    if pred_frames: pd.concat(pred_frames, ignore_index=True).to_csv(args.out / "shortcut_probe_predictions.csv", index=False)
    print(f"[audit] wrote {args.out / 'identity_audit_summary.json'}")
    print(f"[audit] wrote {args.out / 'identity_audit_summary.csv'}")
    if pred_frames: print(f"[audit] wrote {args.out / 'shortcut_probe_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
