#!/usr/bin/env python3
"""Retrospective canine SCC scanner-only A baseline. Does NOT prove pure biology."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

SOURCE_SHA = "78a736bd8c672d9fedb2719db89cc9045567efaff726d67af5029c6d72511e25"
SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")

def load(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SOURCE_SHA:
        raise ValueError("Unrecognized canine source ZIP checksum")
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt ZIP")
        ledger = json.loads(z.read("COLLECTION_MANIFEST.json"))
        index = {item["entry"]: item for item in ledger["included"]}
        def member(suffix):
            matches = [name for name in index if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("Missing/ambiguous ZIP member: " + suffix)
            name = matches[0]
            blob = z.read(name)
            item = index[name]
            if hashlib.sha256(blob).hexdigest().lower() != item["sha256"].lower() or len(blob) != item["bytes"]:
                raise ValueError("Member SHA256 mismatch: " + name)
            return blob
        with np.load(io.BytesIO(member("external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")), allow_pickle=False) as data:
            features = data["features"].copy()
            region = data["region_id"].astype(str)
            scanner = data["scanner_id"].astype(str)
            sample = data["slide_id"].astype(str)
        meta = pd.read_csv(io.BytesIO(member("external_multiscanner_caninescc/geometry_qualified/geometry_qualified_manifest.csv")))
    if features.shape != (4025, 768) or not np.isfinite(features).all():
        raise ValueError("Bad frozen feature array")
    keys = list(zip(region, scanner))
    meta_keys = list(zip(meta.region_id.astype(str), meta.scanner_id.astype(str)))
    if len(set(keys)) != 4025 or len(set(meta_keys)) != 4025 or set(keys) != set(meta_keys):
        raise ValueError("Source/metadata one-to-one key mismatch")
    meta_rows = {key: i for i, key in enumerate(meta_keys)}
    offset = np.array([meta_rows[key] for key in keys])
    fold = meta.fold.to_numpy(dtype=int)[offset]
    category = meta.category_name.astype(str).to_numpy()[offset]
    if not np.array_equal(sample, meta.sample_id.astype(str).to_numpy()[offset]):
        raise ValueError("Sample mismatch")
    regions = np.unique(region)
    if len(regions) != 805 or set(scanner) != set(SCANNERS):
        raise ValueError("Invalid paired-region cohort")
    row = {key: i for i, key in enumerate(keys)}
    grid = np.array([[row[(r, s)] for s in SCANNERS] for r in regions], dtype=int)
    first = grid[:, 0]
    for arm in grid:
        if len(set(sample[arm])) != 1 or len(set(fold[arm])) != 1 or len(set(category[arm])) != 1:
            raise ValueError("Broken biological-unit grouping")
    for case in np.unique(sample):
        if len(set(fold[sample == case])) != 1:
            raise ValueError("Biological sample crosses folds")
    return features[grid].astype(float), fold[first], category[first]

def linear_probe(x, y, train, test):
    scaler = StandardScaler().fit(x[train])
    clf = RidgeClassifier(alpha=100, class_weight="balanced", solver="cholesky")
    clf.fit(scaler.transform(x[train]), y[train])
    return float(balanced_accuracy_score(y[test], clf.predict(scaler.transform(x[test]))))

def run(path, out):
    x, fold, category = load(path)
    acq = np.broadcast_to(np.eye(5)[None, :, :], (805, 5, 5))
    assert all(np.unique(acq[:, s, :], axis=0).shape[0] == 1 for s in range(5))
    scanner = np.tile(np.arange(5), 805)
    tissue = np.repeat(category, 5)
    valid = tissue != "Cartilage"  # Two regions do not support category generalization.
    linear, nonlinear, mse = [], [], []
    for k in range(5):
        train_reg, test_reg = fold != k, fold == k
        means = x[train_reg].mean(axis=0)
        b = x - means[None, :, :]  # B(x,s) = x - mu_s; mu_s fit train-only.
        train = np.repeat(train_reg, 5)
        test = np.repeat(test_reg, 5)
        for task, label in (("scanner", scanner), ("category", tissue)):
            for family, vec in (("raw", x), ("scanner_centered_B", b)):
                score = linear_probe(vec.reshape(-1, 768), label, train & valid, test & valid)
                linear.append({"fold": k, "task": task, "family": family, "balanced_accuracy": score})
        # One fixed-capacity nonlinear challenger, not an upper bound on leakage.
        forest = ExtraTreesClassifier(n_estimators=24, max_depth=12, min_samples_leaf=10,
                                     max_features="sqrt", n_jobs=1, random_state=20261103)
        flat = b.reshape(-1, 768)
        forest.fit(flat[train], scanner[train])
        nonlinear.append(float(balanced_accuracy_score(scanner[test], forest.predict(flat[test]))))
        for source in range(5):
            for target in range(5):
                if source == target:
                    continue
                # Only source features enter predictions; target heldout pixels/features score results.
                prediction = b[test_reg, source, :] + means[target]
                target_true = x[test_reg, target, :]
                mse.append((int(test_reg.sum()), float(np.mean((prediction - target_true) ** 2))))
    import pandas as pd
    frame = pd.DataFrame(linear)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "linear_folds.csv", index=False)
    summary = {
        "status": "RETROSPECTIVE_DEVELOPMENT_NO_PURITY_CLAIM",
        "source_sha256": SOURCE_SHA, "samples": 44, "regions": 805,
        "ordered_scanner_pairs": 20, "pair_predictions": 16100,
        "linear_balanced_accuracy_mean_folds": {
            task + "/" + fam: float(group.balanced_accuracy.mean())
            for (task, fam), group in frame.groupby(["task", "family"])
        },
        "nonlinear_scanner_B_mean": float(np.mean(nonlinear)),
        "nonlinear_scanner_B_folds": nonlinear,
        "mean_shift_mse": float(np.average([v for n, v in mse], weights=[n for n, v in mse])),
        "gates": {"A_instance_channel": "ABSENT_BY_CONSTRUCTION",
                  "B_nonlinear_scanner_invariance": "FAIL",
                  "B_real_biological_purity": "UNPROVEN",
                  "fresh_independent_confirmation": "NOT_ELIGIBLE"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.source_zip, args.out)
