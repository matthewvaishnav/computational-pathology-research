#!/usr/bin/env python3
"""Read-only 256px bridge: original legacy warp vs new tiled path and pinned processor.

Uses the established TRAIN-only historical parity receipt as a *linked source*;
does not re-run SCORPION TRAIN encoder, PA-NF checkpoints or protected TEST.
No 512/1024 representation evaluation is authorized by this bridge alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.external_multiscanner.scancomp_panf_001r_largefield_qc import _source_gate
from scripts.external_multiscanner.scancomp_panf_001r_site_lock import (
    REFERENCE, SLIDE, TARGETS, authenticate_sites, sha256,
)
from scripts.external_multiscanner.scancomp_panf_001r_tiled import (
    read_reference_tiled, warp_affine_tiled,
)
from scripts.external_multiscanner.scancomp_visual_holdout_qc import (
    warp_affine_target_patch,
)

PROCESSOR_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
PROCESSOR_REPO = "facebook/dinov2-base"


def _validate_processor(processor) -> None:
    if (
        type(processor).__name__ != "BitImageProcessor"
        or processor.do_resize is not True
        or processor.size != {"shortest_edge": 256}
        or processor.do_center_crop is not True
        or processor.crop_size != {"height": 224, "width": 224}
        or processor.do_rescale is not True
        or not math.isclose(float(processor.rescale_factor), 1.0 / 255.0, abs_tol=1e-9)
        or processor.do_normalize is not True
        or list(processor.image_mean) != [0.485, 0.456, 0.406]
        or list(processor.image_std) != [0.229, 0.224, 0.225]
        or int(processor.resample) != int(Image.Resampling.BICUBIC)
    ):
        raise ValueError("Cached BitImageProcessor does not match frozen original contract")


def _pixel_hash(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()


def _tensor(processor, rgb: np.ndarray) -> np.ndarray:
    result = processor(images=Image.fromarray(rgb, "RGB"), return_tensors="np")
    tensor = np.asarray(result["pixel_values"][0], dtype=np.float32)
    if tensor.shape != (3, 224, 224) or not np.isfinite(tensor).all():
        raise ValueError("Invalid original-processor tensor")
    return tensor


def run(args: argparse.Namespace) -> dict:
    import cv2
    from transformers import AutoImageProcessor
    from src.data.wsi_pipeline.reader import WSIReader

    receipt = json.loads(args.registration_receipt.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("registration_receipt_sha256") != sha256(args.registration_receipt):
        raise ValueError("Original preflight does not bind registration receipt")
    _source_gate(receipt, preflight, args.data_dir)
    sites, provenance = authenticate_sites(
        args.sites, args.registration_receipt, args.landmarks, args.visual_receipt
    )
    historical = json.loads(args.historical_train_parity.read_text(encoding="utf-8"))
    if not isinstance(historical, dict):
        raise ValueError("Historical TRAIN-only parity receipt is not a JSON object")
    # Do not derive acceptance from unknown historical receipt field names.
    processor = AutoImageProcessor.from_pretrained(
        PROCESSOR_REPO, revision=PROCESSOR_REVISION, use_fast=False, local_files_only=True
    )
    _validate_processor(processor)
    observations = []
    with ExitStack() as stack:
        readers = {
            scanner: stack.enter_context(
                WSIReader(args.data_dir / receipt["source_files"][scanner]["filename"])
            ) for scanner in (REFERENCE, *TARGETS)
        }
        for site in sites:
            x = int(site["center_x_px"] - 128)
            y = int(site["center_y_px"] - 128)
            old_ref = readers[REFERENCE].read_region((x, y), 0, (256, 256))
            new_ref, _tiles = read_reference_tiled(readers[REFERENCE], x, y, 256, 256)
            for scanner in (REFERENCE, *TARGETS):
                if scanner == REFERENCE:
                    old, new = old_ref, new_ref
                else:
                    affine = receipt["scanners"][scanner]["models"]["affine"]
                    old = warp_affine_target_patch(readers[scanner], affine, x, y, 256)
                    new, _tiles = warp_affine_tiled(readers[scanner], affine, x, y, 256, 256)
                if old.shape != (256, 256, 3) or new.shape != old.shape:
                    raise ValueError("256px old/new source shape mismatch")
                difference = np.abs(old.astype(np.int16) - new.astype(np.int16))
                old_tensor, new_tensor = _tensor(processor, old), _tensor(processor, new)
                tensor_delta = new_tensor.astype(np.float64) - old_tensor.astype(np.float64)
                tensor_max = float(np.abs(tensor_delta).max())
                tensor_rel = float(
                    np.linalg.norm(tensor_delta) / max(np.linalg.norm(old_tensor), 1e-12)
                )
                observations.append({
                    "site_id": site["site_id"], "scanner": scanner,
                    "source_top_left_xy": [x, y],
                    "legacy_rgb_sha256": _pixel_hash(old),
                    "tiled_rgb_sha256": _pixel_hash(new),
                    "rgb_max_abs_difference": int(difference.max()),
                    "processor_max_abs_difference": tensor_max,
                    "processor_relative_l2_difference": tensor_rel,
                })
    max_rgb = max(v["rgb_max_abs_difference"] for v in observations)
    max_tensor = max(v["processor_max_abs_difference"] for v in observations)
    max_relative = max(v["processor_relative_l2_difference"] for v in observations)
    passed = max_rgb <= 1 and max_tensor <= 0.02 and max_relative <= 0.001
    record = {
        "stage": "SCANCOMP_PANF_001R_ORIGINAL_PROCESSOR_256PX_BRIDGE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slide_alias": SLIDE, "site_provenance": provenance,
        "source_sha256_verified": True,
        "original_registration_receipt_sha256": sha256(args.registration_receipt),
        "historical_train_only_parity_receipt_sha256": sha256(args.historical_train_parity),
        "historical_train_only_parity_reexecution_performed": False,
        "direct_original_pilot_saved_rgb_bytes_compared": False,
        "geometry_bridge_is_legacy_helper_not_original_pilot_byte_replay": True,
        "encoder_or_checkpoint_run": False,
        "processor": {"repo": PROCESSOR_REPO, "revision": PROCESSOR_REVISION,
                      "class": type(processor).__name__,
                      "normalization": "frozen ImageNet RGB"},
        "observations_expected": 32, "observations": observations,
        "max_rgb_abs_difference": max_rgb,
        "max_processor_abs_difference": max_tensor,
        "max_processor_relative_l2_difference": max_relative,
        "bridge_pass": passed,
        "inference_admitted": False,
    }
    if len(observations) != 32 or not passed:
        raise ValueError("256px tiled/legacy preprocessor parity did not pass")
    if args.out.exists():
        raise ValueError("Refusing to replace existing parity receipt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + ".pending")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.out)
    return {"receipt": str(args.out), "observations": 32,
            "bridge_pass": True, "inference_admitted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registration-receipt", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--visual-receipt", type=Path, required=True)
    parser.add_argument("--historical-train-parity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print("256PX PILOT BRIDGE FAILED CLOSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
