#!/usr/bin/env python3
"""Extract centered 96px DINOv2 features for ALL 1,614 MITOS candidate pairs.

Development cohort only. This program DOES NOT train PA-NF, select a
hyperparameter, or evaluate on independent physical slides.
Writes to a staged directory; finalize/rename after the Python process exits
to avoid locked Windows memory-map file handles.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
REPO = "facebook/dinov2-base"
CROP = 96
OFFSET = (224 - CROP) // 2
SOURCE_HASHES = {
    "A_rgb_uint8.npy": "BD54F980B2F98DB75D7F46B7347FCA3B1B16B9183D217FB54B3CAEC74BC5DF76",
    "H_aligned_rgb_uint8.npy": "4A60F9078138DC383B8ED41E497093479918755989514D39D152969F8431BE39",
    "patch_index.jsonl": "6321217CD9023A3E99538A731F5689FDAA30F5B5ABF8238FDB83C742B7318178",
}
SMOKE_SHA = "5A646E634B6814D33156292EBD710C8EA1E5E4BBEAE758D6DD8C6A40361AD365"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_records(path: Path) -> list[dict]:
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1614 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected or duplicate row inventory")
    for i, row in enumerate(rows):
        if row["row"] != i or row["slide"] not in ("07", "17"):
            raise ValueError(f"Misordered/invalid record at row {i}")
        if row["source"] not in ("grid", "annotated"):
            raise ValueError("Unexpected source class")
        if row["annotation_class"] not in (None, "mitosis", "not_mitosis"):
            raise ValueError("Unexpected annotation label")
    return rows


def crop_center(arr: np.ndarray, row: int) -> np.ndarray:
    patch = np.asarray(arr[row, OFFSET:OFFSET+CROP, OFFSET:OFFSET+CROP, :])
    if patch.shape != (CROP, CROP, 3) or patch.dtype != np.uint8:
        raise ValueError(f"Invalid 96px crop at row {row}")
    return np.ascontiguousarray(patch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch-size must be in 1..32")

    root = args.downloads
    source = root / "PA-NF-MITOS-PAIRED-PATCHES-v1"
    smoke_path = root / "PA-NF-MITOS-DINOV2-SMOKE-v1" / "smoke_result.json"
    staged = root / "PA-NF-MITOS-ALL96-DINOV2-v1.incomplete"
    final = root / "PA-NF-MITOS-ALL96-DINOV2-v1"
    if staged.exists() or final.exists():
        raise RuntimeError("Prior output exists; refusing overwrite")

    for name, expected in SOURCE_HASHES.items():
        if sha256(source / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
        print("SOURCE PASS:", name, flush=True)
    if sha256(smoke_path) != SMOKE_SHA:
        raise RuntimeError("Pinned smoke-test report mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke["revision"] != REVISION or smoke["resolved_revision"] != REVISION:
        raise RuntimeError("Smoke-test revision mismatch")

    rows = load_records(source / "patch_index.jsonl")
    a = np.load(source / "A_rgb_uint8.npy", mmap_mode="r")
    h = np.load(source / "H_aligned_rgb_uint8.npy", mmap_mode="r")
    if (a.shape != h.shape or a.shape != (1614, 224, 224, 3)
            or a.dtype != np.uint8 or h.dtype != np.uint8):
        raise RuntimeError("Source patch arrays differ from frozen contract")

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no output written")
    processor = AutoImageProcessor.from_pretrained(REPO, revision=REVISION)
    proc_text = json.dumps(processor.to_dict(), sort_keys=True, default=str)
    proc_sha = hashlib.sha256(proc_text.encode("utf-8")).hexdigest().upper()
    if (proc_sha != smoke["processor_config_sha256"]
            or type(processor).__name__ != smoke["processor"]):
        raise RuntimeError("DINOv2 preprocessing differs from smoke test")

    model = AutoModel.from_pretrained(REPO, revision=REVISION).eval()
    if str(getattr(model.config, "_commit_hash", None)) != REVISION:
        raise RuntimeError("Model revision does not match frozen revision")
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad_(False)

    staged.mkdir()
    feature_path = staged / "all96_cls_float32.npy"
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(1614, 2, 768)
    )
    with torch.inference_mode():
        for name, arm, arr in (("A", 0, a), ("H", 1, h)):
            for begin in range(0, len(rows), args.batch_size):
                end = min(begin + args.batch_size, len(rows))
                images = [Image.fromarray(crop_center(arr, i)) for i in range(begin, end)]
                px = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(pixel_values=px).last_hidden_state[:, 0, :]
                batch = output.float().cpu().numpy()
                if batch.shape != (end-begin, 768) or not np.isfinite(batch).all():
                    raise RuntimeError(f"Invalid embedding: {name}/{begin}:{end}")
                features[begin:end, arm] = batch
                if end % 256 < args.batch_size or end == len(rows):
                    print(f"{name}: {end}/{len(rows)}", flush=True)
            features.flush()
    del features
    gc.collect()

    idx = staged / "all96_index.jsonl"
    with idx.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({
                "row": r["row"], "id": r["id"], "slide": r["slide"],
                "field": r["field"], "source": r["source"],
                "annotation_class": r["annotation_class"],
                "annotation_correspondence_20px": r["annotation_correspondence_20px"],
                "A_tissue_fraction_heuristic": r["A_tissue_fraction_heuristic"],
                "H_tissue_fraction_heuristic": r["H_tissue_fraction_heuristic"],
                "grayscale_corr_descriptive": r["grayscale_corr_descriptive"],
                "patient_id": r["patient_id"],
            }, allow_nan=False) + "\n")

    # Reconcile every previously materialized annotation-centered embedding.
    previous = root / "PA-NF-MITOS-CELL-CENTERED-v1"
    prior_report = previous / "localized_pilot_report.json"
    prior_digest = "BDDDB702BC1EB739B41B40EC172E954D0925266108134EE5E0C19E04B3CEB882"
    if sha256(prior_report) != prior_digest:
        raise RuntimeError("Prior 392-pair localized report digest mismatch")
    prev = json.loads(prior_report.read_text(encoding="utf-8"))
    for name, expected in prev["output_hashes"].items():
        if sha256(previous / name) != expected:
            raise RuntimeError("Prior localized source hash mismatch: " + name)
    oldrows = [json.loads(s) for s in
               (previous / "cell_centered_index.jsonl").read_text(encoding="utf-8").splitlines()]
    oldfeatures = np.load(previous / "cell_centered_cls_float32.npy")
    fullfeatures = np.load(feature_path, mmap_mode="r")
    if len(oldrows) != 392 or oldfeatures.shape != (392, 2, 768):
        raise RuntimeError("Unexpected prior localized inventory")
    for i, row in enumerate(oldrows):
        if (row["localized_row"] != i or
            row["id"] != rows[row["original_row"]]["id"]):
            raise RuntimeError("Localized identity mismatch")
    observed = fullfeatures[np.array([r["original_row"] for r in oldrows])]
    aa = oldfeatures.astype(np.float64)
    bb = np.asarray(observed, dtype=np.float64)
    cosine = (aa*bb).sum(axis=-1) / (
        np.linalg.norm(aa,axis=-1)*np.linalg.norm(bb,axis=-1))
    parity = {"min_cosine": float(cosine.min()),
              "max_abs_difference": float(np.max(np.abs(aa-bb)))}
    if not np.isfinite(cosine).all() or parity["min_cosine"] < 0.9999:
        raise RuntimeError(f"392-pair extraction parity failed: {parity}")
    del fullfeatures, observed
    print("PASS: 392 prior localized pairs reproduced", parity, flush=True)

    report = {
        "localized_392_report_sha256": prior_digest,
        "localized_392_parity": parity,
        "status": "STAGED_FULL_96PX_EMBEDDINGS_NO_TRAINING",
        "shape": [1614, 2, 768],
        "crop_bounds": [OFFSET, OFFSET+CROP],
        "source": "central 96x96 of registered 224x224 pair; grid crops are NOT cell-annotated",
        "model": REPO, "model_revision": REVISION,
        "processor_class": type(processor).__name__,
        "processor_config_sha256": proc_sha,
        "precision": "CUDA autocast float16, float32 stored",
        "source_hashes": SOURCE_HASHES,
        "smoke_report_sha256": SMOKE_SHA,
        "output_sha256": {
            feature_path.name: sha256(feature_path),
            idx.name: sha256(idx),
        },
        "scope": "Only two physical slides; development, not independent confirmation",
    }
    rp = staged / "all96_report.json"
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== ALL96 FEATURE EXTRACTION STAGED ===")
    print(json.dumps(report, indent=2))
    print("Report SHA256:", sha256(rp))
    print("EXIT PYTHON BEFORE RENAMING:", staged, "->", final)


if __name__ == "__main__":
    main()
