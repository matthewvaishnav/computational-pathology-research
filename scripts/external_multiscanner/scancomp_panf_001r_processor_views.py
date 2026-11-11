#!/usr/bin/env python3
"""Verify the pinned slow processor's actual tensors on QC-only 512/1024px RGB.

No encoder/checkpoints are loaded. Checks PNG/receipt byte integrity, verifies
224x224 visible pixels against tensor normalization and saves read-only receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.external_multiscanner.scancomp_panf_001r_largefield_qc import (
    _processor_geometry_view,
)
from scripts.external_multiscanner.scancomp_panf_001r_pilot_bridge import (
    PROCESSOR_REPO, PROCESSOR_REVISION, _tensor, _validate_processor,
)
from scripts.external_multiscanner.scancomp_panf_001r_site_lock import (
    LOCKED_SHA256, sha256,
)


def _normalize_geometry_view(preview: Image.Image) -> np.ndarray:
    rgb = np.asarray(preview.convert("RGB"), dtype=np.float32)
    arr = rgb / np.float32(255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose((arr - mean) / std, (2, 0, 1))


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoImageProcessor

    qc = json.loads(args.qc_receipt.read_text(encoding="utf-8"))
    if (
        qc.get("stage") != "SCANCOMP_PANF_001R_LARGEFIELD_FIT_ONLY_VISUAL_QC"
        or qc.get("site_provenance", {}).get("site_manifest_sha256") != LOCKED_SHA256
        or qc.get("model_inference_performed") is not False
        or qc.get("site_count") != 8
        or qc.get("observations_expected") != 64
        or len(qc.get("fields", [])) != 64
    ):
        raise ValueError("Incomplete/untrusted large-field QC receipt")
    processor = AutoImageProcessor.from_pretrained(
        PROCESSOR_REPO, revision=PROCESSOR_REVISION, use_fast=False, local_files_only=True
    )
    _validate_processor(processor)
    prepared = []
    for item in qc["fields"]:
        name = item["target_full_png"]
        native_path = args.qc_receipt.parent / name
        if native_path.name != name or sha256(native_path) != item["target_full_png_sha256"]:
            raise ValueError("Native full-field PNG provenance mismatch: " + str(name))
        with Image.open(native_path) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        side = item["reference_box_xywh"][2]
        if rgb.shape != (side, side, 3) or side not in (512, 1024):
            raise ValueError("Unexpected native RGB shape/size: " + name)
        preview = _processor_geometry_view(rgb)
        old_preview = args.qc_receipt.parent / item["target_processor_view"]
        if (
            old_preview.name != item["target_processor_view"]
            or sha256(old_preview) != item["target_processor_view_sha256"]
        ):
            raise ValueError("Processor-geometry preview provenance mismatch")
        with Image.open(old_preview) as view:
            if not np.array_equal(np.asarray(view.convert("RGB")), np.asarray(preview)):
                raise ValueError("Processor-view PNG changed since QC")
        tensor = _tensor(processor, rgb)
        geom = _normalize_geometry_view(preview)
        delta = tensor.astype(np.float64) - geom.astype(np.float64)
        mx = float(np.abs(delta).max())
        rel = float(np.linalg.norm(delta) / max(np.linalg.norm(tensor), 1e-12))
        if mx > 0.02 or rel > 0.001:
            raise ValueError("Processor tensor differs from saved 224px visible context")
        prepared.append({
            "site_id": item["site_id"], "scanner": item["scanner"],
            "reference_side_pixels": side, "native_png": name,
            "native_png_sha256": item["target_full_png_sha256"],
            "preview_png": item["target_processor_view"],
            "preview_png_sha256": item["target_processor_view_sha256"],
            "pinned_processor_tensor_sha256": hashlib.sha256(
                np.ascontiguousarray(tensor, dtype=np.float32).tobytes()
            ).hexdigest(),
            "max_normalized_pixel_delta": mx, "relative_l2_pixel_delta": rel,
            "tensor_shape": [3, 224, 224],
        })
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing processor parity receipt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "stage": "SCANCOMP_PANF_001R_LARGEFIELD_PROCESSOR_PIXEL_PARITY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_largefield_qc_sha256": sha256(args.qc_receipt),
        "processor": PROCESSOR_REPO, "revision": PROCESSOR_REVISION,
        "input_fields": len(prepared), "pinned_preprocessing_visual_parity_pass": True,
        "model_execution_performed": False, "inference_admitted": False,
        "records": prepared,
    }
    pending = args.out.with_name(args.out.name + ".pending")
    pending.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(args.out)
    return {"receipt": str(args.out), "fields": len(prepared),
            "processor_pixel_parity_pass": True, "inference_admitted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print("LARGE-FIELD PROCESSOR PARITY FAILED CLOSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
