#!/usr/bin/env python3
"""Two-slide MITOS exploratory PA-NF versus equal-capacity neutral two-branch control.

Uses ALL 1,614 96px paired DINOv2 regions for unsupervised training, not only
annotation-centered crops. Each fold fits on one physical slide and never
backpropagates through the other. Pre-registered fixed epochs/seeds, no tuning.
This is a DEVELOPMENT pilot, NOT independent clinical or patient-level evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.models.scorpion_pathoalign import (  # noqa: E402
    ProjectionConfig, ScorpionProjection, projection_loss,
)

SEEDS = (801, 802, 803)
EPOCHS = 75
REGION_BATCH = 32
LR = 3e-4
WD = 1e-4
PROBE_L2 = 0.01
SOURCE_MODEL = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_CONFIG = ProjectionConfig(
    input_dim=768, biological_dim=256, acquisition_dim=64,
    hidden_dim=512, temperature=0.10,
    scanner_adversary_weight=0.5, scanner_acquisition_weight=0.5,
    scanner_dependence_weight=20.0, cross_covariance_weight=0.05,
    gradient_reversal_strength=1.0,
)
NEUTRAL_CONFIG = replace(
    MODEL_CONFIG, scanner_adversary_weight=0.0,
    scanner_acquisition_weight=0.0, scanner_dependence_weight=0.0,
    cross_covariance_weight=0.0, gradient_reversal_strength=0.0,
)
METHODS = {"neutral_two_branch": NEUTRAL_CONFIG, "panf": MODEL_CONFIG}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest().upper()


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < 1e-10):
        raise ValueError("Feature norm invalid or collapsed to zero")
    return x / norm


def auc_ap(y: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    p = int(y.sum())
    n = len(y) - p
    if min(p, n) < 1 or not np.isfinite(scores).all():
        raise ValueError("AUROC/AP requires both classes and finite scores")
    ranks = rankdata(scores, method="average")
    auc = float((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n))
    order = np.argsort(-scores, kind="mergesort")
    sy = y[order]
    precision = np.cumsum(sy) / np.arange(1, len(y)+1)
    ap = float(np.mean(precision[sy == 1]))
    return {"n": len(y), "positives": p, "AUROC": auc, "AP": ap,
            "prevalence": p / len(y)}


def fit_logistic(z: np.ndarray, y: np.ndarray, *, balanced: bool = True):
    y = np.asarray(y, dtype=float)
    p = int(y.sum())
    n = len(y)-p
    if min(p, n) < 5:
        raise ValueError("Need >=5 of each class in FIT")
    weights = (np.where(y == 1, 0.5/p, 0.5/n)
               if balanced else np.full(len(y), 1/len(y)))
    def loss(beta):
        w, b = beta[:-1], beta[-1]
        t = z @ w + b
        v = np.sum(weights * (np.logaddexp(0, t) - y*t)) + PROBE_L2/2 * (w @ w)
        grad_t = weights * (expit(t) - y)
        return float(v), np.r_[z.T @ grad_t + PROBE_L2*w, grad_t.sum()]
    fit = minimize(loss, np.zeros(z.shape[1]+1), jac=True, method="L-BFGS-B",
                   options={"maxiter":500, "ftol":1e-12})
    if not fit.success and np.linalg.norm(fit.jac) > 1e-5:
        raise RuntimeError("Probe optimizer failed: " + str(fit.message))
    return fit.x[:-1], float(fit.x[-1])


def retrieval(b: np.ndarray, records: list[dict], selected: np.ndarray) -> dict:
    """Exact-row retrieval among SAME-FIELD test candidates, both directions."""
    fields = {}
    for row in selected:
        fields.setdefault(records[int(row)]["field"], []).append(int(row))
    ah = ha = n = 0
    for group in fields.values():
        if len(group) < 2:
            continue
        sim = b[group, 0] @ b[group, 1].T
        target = np.arange(len(group))
        ah += int((sim.argmax(axis=1) == target).sum())
        ha += int((sim.argmax(axis=0) == target).sum())
        n += len(group)
    return {"eligible": n, "A_to_H_hits": ah, "H_to_A_hits": ha,
            "A_to_H_recall1": ah/n if n else None,
            "H_to_A_recall1": ha/n if n else None}


def outcomes(b: np.ndarray, a: np.ndarray | None,
             records: list[dict], fit_rows: np.ndarray,
             heldout_rows: np.ndarray) -> dict:
    """FIT-only label/scanner probes; score TEST without retraining."""
    b = unit(b.astype(np.float64))
    labeled_fit = np.array([i for i in fit_rows
                            if records[i]["annotation_class"] in ("mitosis", "not_mitosis")])
    labeled_test = np.array([i for i in heldout_rows
                             if records[i]["annotation_class"] in ("mitosis", "not_mitosis")])
    yf = np.array([records[int(i)]["annotation_class"] == "mitosis"
                   for i in labeled_fit], dtype=int)
    yt = np.array([records[int(i)]["annotation_class"] == "mitosis"
                   for i in labeled_test], dtype=int)
    w, c = fit_logistic(b[labeled_fit, 0], yf)
    label_results = {
        "heldout_A": auc_ap(yt, b[labeled_test, 0] @ w + c),
        "heldout_H": auc_ap(yt, b[labeled_test, 1] @ w + c),
    }
    corr_mask = np.array([records[int(i)]["annotation_correspondence_20px"] is True
                          for i in labeled_test], dtype=bool)
    if corr_mask.sum() and 0 < yt[corr_mask].sum() < corr_mask.sum():
        label_results["heldout_H_annotation_correspondence_20px"] = auc_ap(
            yt[corr_mask], b[labeled_test[corr_mask], 1] @ w + c
        )

    train_views = b[fit_rows].reshape(-1, b.shape[2])
    test_views = b[heldout_rows].reshape(-1, b.shape[2])
    scanner_train = np.tile([0, 1], len(fit_rows))
    scanner_test = np.tile([0, 1], len(heldout_rows))
    ws, bs = fit_logistic(train_views, scanner_train)
    predicted = (test_views @ ws + bs >= 0).astype(int)
    scanner_bal_acc = 0.5 * (
        np.mean(predicted[scanner_test == 0] == 0) +
        np.mean(predicted[scanner_test == 1] == 1)
    )
    result = {
        "mitosis_probe_train_on_A_only": {
            "fit_positives": int(yf.sum()), "fit_negatives": int(len(yf)-yf.sum()),
            "L2": PROBE_L2, "results": label_results,
        },
        "scanner_probe_fit_to_heldout_balanced_accuracy": float(scanner_bal_acc),
        "retrieval": retrieval(b, records, heldout_rows),
        "biological_feature_dim": b.shape[2],
        "biological_variance_mean_heldout": float(b[heldout_rows].var(axis=(0,1)).mean()),
    }
    if a is not None:
        acq = unit(a.astype(np.float64))
        zs = acq[fit_rows].reshape(-1, acq.shape[-1])
        ws, bs = fit_logistic(zs, scanner_train)
        pred = (acq[heldout_rows].reshape(-1, acq.shape[-1]) @ ws + bs >= 0).astype(int)
        result["acquisition_scanner_probe_balanced_accuracy"] = float(
            0.5*(np.mean(pred[scanner_test == 0] == 0) +
                 np.mean(pred[scanner_test == 1] == 1))
        )
    return result


def run_one(features, records, train, test, config, method, seed, epochs,
            device, output_dir):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mu = features[train].mean(axis=(0,1), keepdims=True)
    sd = features[train].std(axis=(0,1), keepdims=True)
    sd = np.where(sd < 1e-6, 1, sd)
    normalized = ((features - mu) / sd).astype(np.float32)
    model = ScorpionProjection("pathoalign", config, n_scanners=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train)
        for begin in range(0, len(shuffled), REGION_BATCH):
            pairrows = shuffled[begin:begin+REGION_BATCH]
            if len(pairrows) < 2:
                continue
            inp = torch.from_numpy(normalized[pairrows].reshape(-1,768)).to(device)
            scan = torch.tensor(np.tile([0,1],len(pairrows)),
                                dtype=torch.long, device=device)
            rid = torch.tensor(np.repeat(np.arange(len(pairrows)),2),
                               dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss, _ = projection_loss(model, inp, scan, rid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {method}/{seed}/{epoch}")
            loss.backward()
            nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(nnorm):
                raise RuntimeError("Nonfinite gradient norm")
            opt.step()
        if epoch == 0 or epoch+1 in (25,50,epochs):
            print(f"{method} seed={seed} epoch={epoch+1}/{epochs} "
                  f"loss={float(loss.detach().cpu()):.6f}", flush=True)

    model.eval()
    bio = np.empty((len(records),2,config.biological_dim),np.float32)
    acq = np.empty((len(records),2,config.acquisition_dim),np.float32)
    with torch.inference_mode():
        for begin in range(0,len(records),256):
            end = min(begin+256,len(records))
            t = torch.from_numpy(normalized[begin:end].reshape(-1,768)).to(device)
            pred = model(t)
            bio[begin:end] = pred["biological"].cpu().numpy().reshape(-1,2,config.biological_dim)
            acq[begin:end] = pred["acquisition"].cpu().numpy().reshape(-1,2,config.acquisition_dim)
    if not np.isfinite(bio).all() or not np.isfinite(acq).all():
        raise RuntimeError("Nonfinite projection")
    result = outcomes(bio,acq,records,train,test)
    output_dir.mkdir(parents=True)
    import torch
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "seed":seed,"epochs":epochs,"method":method,
                "config":asdict(config),"fit_slide":records[int(train[0])]["slide"],
                "mean":mu,"std":sd,"encoder_revision":SOURCE_MODEL},
               output_dir/"checkpoint.pt")
    np.savez_compressed(output_dir/"projected_features.npz",biological=bio,acquisition=acq)
    result["artifact_sha256"] = {
        "checkpoint.pt":sha256(output_dir/"checkpoint.pt"),
        "projected_features.npz":sha256(output_dir/"projected_features.npz"),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path,default=Path.home()/"Downloads")
    parser.add_argument("--smoke",action="store_true",
                        help="One seed, one slide fold, one epoch; never evidence")
    args=parser.parse_args()
    import torch

    root=args.downloads
    source=root/"PA-NF-MITOS-ALL96-DINOV2-v1"
    prev=root/"PA-NF-MITOS-DINOV2-FEATURES-v1"/"feature_report.json"
    prevr="90D620E172FC43992C79F51B3D630869043CA18FD755634B668CB643A18CD462"
    if sha256(prev)!=prevr:
        raise RuntimeError("Original 224px feature report mismatch")
    frozen=json.loads((source/"all96_report.json").read_text(encoding="utf-8"))
    if (frozen["status"]!="STAGED_FULL_96PX_EMBEDDINGS_NO_TRAINING"
        or frozen["model_revision"]!=SOURCE_MODEL or frozen["shape"]!=[1614,2,768]):
        raise RuntimeError("Unexpected all96 feature report")
    for name,expected in frozen["output_sha256"].items():
        if sha256(source/name)!=expected:
            raise RuntimeError("All96 content hash mismatch: "+name)
    feature=np.load(source/"all96_cls_float32.npy")
    if feature.shape!=(1614,2,768) or not np.isfinite(feature).all():
        raise RuntimeError("Invalid all96 feature array")
    records=[json.loads(s) for s in (source/"all96_index.jsonl").read_text().splitlines()]
    if (len(records)!=1614 or len({r["id"] for r in records})!=1614
        or any(r["row"]!=i for i,r in enumerate(records))):
        raise RuntimeError("Unexpected all96 index")

    planned=[("07","17")] if args.smoke else [("07","17"),("17","07")]
    seeds=SEEDS[:1] if args.smoke else SEEDS
    epochs=1 if args.smoke else EPOCHS
    staged=root/("PA-NF-MITOS-PROJECTION-SMOKE-v1.incomplete" if args.smoke
                 else "PA-NF-MITOS-PROJECTION-PILOT-v1.incomplete")
    final=root/("PA-NF-MITOS-PROJECTION-SMOKE-v1" if args.smoke
                else "PA-NF-MITOS-PROJECTION-PILOT-v1")
    if staged.exists() or final.exists():
        raise RuntimeError("Existing output preserved")
    staged.mkdir()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result={"status":"SMOKE_NOT_EVIDENCE" if args.smoke else "EXPLORATORY_TWO_SLIDE_PILOT",
            "model_revision":SOURCE_MODEL,"input_report_sha256":sha256(source/"all96_report.json"),
            "design":{"seeds":list(seeds),"epochs":epochs,"region_batch_size":REGION_BATCH,
                      "learning_rate":LR,"weight_decay":WD,"probe_L2":PROBE_L2,
                      "methods":{k:asdict(v) for k,v in METHODS.items()},
                      "fit_test_slides":planned,"scanners":["Aperio","Hamamatsu"]},
            "limitations":["Two slides, no patient hierarchy", "Retrospective 96px crop selected after 224px failure",
                           "A-origin labels, overlap/nonindependence, no independent confirmation",
                           "No held-out tuning permitted"],"folds":{}}
    for fit_slide,test_slide in planned:
        train=np.flatnonzero(np.array([r["slide"]==fit_slide for r in records]))
        test=np.flatnonzero(np.array([r["slide"]==test_slide for r in records]))
        if len(train)+len(test)!=len(records) or min(len(train),len(test))<100:
            raise RuntimeError("Invalid physical slide split")
        fold=f"train_{fit_slide}_test_{test_slide}"
        raw_unit = unit(feature.astype(np.float64))
        mean_a = raw_unit[train, 0].mean(axis=0)
        mean_h = raw_unit[train, 1].mean(axis=0)
        shifted = raw_unit.copy()
        shifted[:, 1] = unit(raw_unit[:, 1] - mean_h + mean_a)
        result["folds"][fold]={"fit_n":len(train),"test_n":len(test),
                              "raw":outcomes(feature,None,records,train,test),
                              "fit_only_mean_shift":outcomes(shifted,None,records,train,test),
                              "methods":{}}
        for method,config in METHODS.items():
            result["folds"][fold]["methods"][method]={}
            for seed in seeds:
                target=staged/fold/method/f"seed_{seed}"
                value=run_one(feature,records,train,test,config,method,seed,epochs,device,target)
                result["folds"][fold]["methods"][method][str(seed)]=value
                print(f"PASS: {fold} {method} seed {seed} "
                      f"AUC-H={value['mitosis_probe_train_on_A_only']['results']['heldout_H']['AUROC']:.4f}",
                      flush=True)
    rp=staged/"pilot_report.json"
    rp.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("\n=== MITOS PA-NF PROJECTION STAGED ===")
    print("Status:",result["status"])
    print("Report:",rp,"SHA256:",sha256(rp))
    print("Exit Python before renaming the .incomplete directory.")


if __name__ == "__main__":
    main()
