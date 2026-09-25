#!/usr/bin/env python3
"""Build frozen SICAPv2 Phikon bags for external TransnnMIL transport evaluation.

No model outcome is read here. The script validates the frozen 155-WSI / 95-patient
mapping, excludes only the ten preregistered duplicate regions, uses the replay-
validated local Phikon snapshot, and writes one HDF5 feature bag per SICAPv2 WSI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTModel


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)
DEFAULT_REPLAY = Path(
    "experiments/transnnmil/external/phikon_replay_verification_20260925.json"
)
DEFAULT_LABELS = Path(
    "experiments/transnnmil/external/sicapv2_image_labels_seggini_1c90f832.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--replay-report", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\sicapv2_external\phikon_bags"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_indices(wsi_id: str, n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    digest = hashlib.sha256(wsi_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed + offset)
    chosen = np.sort(rng.choice(n, size=cap, replace=False))
    return chosen.astype(np.int64)


def find_images_dir(root: Path) -> Path:
    candidates = [
        root / "images",
        root / "SICAPv2" / "images",
    ]
    hits = [path for path in candidates if path.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one SICAPv2 images directory under {root}; found {hits}"
        )
    return hits[0]


def main() -> None:
    args = parse_args()
    spec = load_json(args.spec)
    replay = load_json(args.replay_report)
    if replay.get("status") != "passed":
        raise RuntimeError("Phikon PANDA replay gate has not passed")
    snapshot = Path(str(replay["snapshot_path"]))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"replay-validated Phikon snapshot is unavailable: {snapshot}")

    mapping = pd.read_csv(args.labels, dtype={"image_id": str, "patient_id": str})
    required = {"image_id", "patient_id", "gleason_score"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"frozen SICAP mapping missing columns: {sorted(missing)}")
    if mapping["image_id"].duplicated().any():
        raise ValueError("frozen SICAP mapping has duplicate image_id rows")

    dataset_spec = spec["external_dataset"]
    if len(mapping) != int(dataset_spec["expected_wsi_units"]):
        raise ValueError(
            f"frozen SICAP mapping has {len(mapping)} WSI rows; "
            f"expected {dataset_spec['expected_wsi_units']}"
        )
    if mapping["patient_id"].nunique() != int(dataset_spec["expected_patients"]):
        raise ValueError(
            f"frozen SICAP mapping has {mapping['patient_id'].nunique()} patients; "
            f"expected {dataset_spec['expected_patients']}"
        )

    target_mapping = {str(k): int(v) for k, v in spec["target_mapping"].items()}
    unknown_scores = sorted(set(mapping["gleason_score"].astype(str)) - set(target_mapping))
    if unknown_scores:
        raise ValueError(f"unmapped SICAP Gleason scores: {unknown_scores}")
    mapping["isup_grade"] = mapping["gleason_score"].astype(str).map(target_mapping).astype(int)

    images_dir = find_images_dir(args.dataset_root)
    image_paths = sorted(images_dir.glob("*.jpg"))
    if len(image_paths) != int(dataset_spec["expected_patch_images"]):
        raise ValueError(
            f"SICAPv2 patch inventory has {len(image_paths)} JPG files; "
            f"expected {dataset_spec['expected_patch_images']}"
        )

    duplicate_regions = tuple(dataset_spec["duplicate_regions_excluded"])
    by_wsi: dict[str, list[Path]] = {image_id: [] for image_id in mapping["image_id"].astype(str)}
    excluded_duplicate_rows: list[dict[str, str]] = []
    unknown_wsi: set[str] = set()

    for path in image_paths:
        stem = path.stem
        wsi_id = stem.split("_", 1)[0]
        if wsi_id not in by_wsi:
            unknown_wsi.add(wsi_id)
            continue
        duplicate = next(
            (region for region in duplicate_regions if stem.startswith(region + "_")),
            None,
        )
        if duplicate is not None:
            excluded_duplicate_rows.append(
                {"wsi_id": wsi_id, "patch": path.name, "duplicate_region": duplicate}
            )
            continue
        by_wsi[wsi_id].append(path)

    if unknown_wsi:
        raise ValueError(f"patches map to unknown SICAP WSI IDs: {sorted(unknown_wsi)}")
    empty = sorted(wsi_id for wsi_id, paths in by_wsi.items() if not paths)
    if empty:
        raise ValueError(f"SICAP WSI units with no eligible patches: {empty}")

    bag_spec = spec["feature_transport"]["bagging"]
    cap = int(bag_spec["max_patches"])
    seed = 20260925
    inventory_rows = []
    for row in mapping.itertuples(index=False):
        wsi_id = str(row.image_id)
        paths = sorted(by_wsi[wsi_id], key=lambda path: path.name)
        chosen = stable_indices(wsi_id, len(paths), cap, seed)
        inventory_rows.append(
            {
                "image_id": wsi_id,
                "patient_id": str(row.patient_id),
                "gleason_score": str(row.gleason_score),
                "isup_grade": int(row.isup_grade),
                "source_patch_count_after_duplicate_exclusion": len(paths),
                "selected_patch_count": len(chosen),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(inventory_rows).sort_values("image_id").reset_index(drop=True)
    inventory_path = args.output_root / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    duplicates_path = args.output_root / "excluded_duplicate_patches.csv"
    pd.DataFrame(excluded_duplicate_rows).to_csv(duplicates_path, index=False)

    if args.inventory_only:
        payload = {
            "status": "inventory_passed",
            "patch_count": len(image_paths),
            "wsi_count": len(inventory),
            "patient_count": inventory["patient_id"].nunique(),
            "duplicate_patch_exclusion_count": len(excluded_duplicate_rows),
            "inventory_sha256": sha256(inventory_path),
        }
        (args.output_root / "inventory_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    device = torch.device(args.device)
    model = ViTModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        add_pooling_layer=False,
    ).to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    feature_dir = args.output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for number, row in enumerate(mapping.sort_values("image_id").itertuples(index=False), 1):
        wsi_id = str(row.image_id)
        all_paths = sorted(by_wsi[wsi_id], key=lambda path: path.name)
        selected_indices = stable_indices(wsi_id, len(all_paths), cap, seed)
        selected_paths = [all_paths[int(index)] for index in selected_indices]
        output_path = feature_dir / f"{wsi_id}.h5"

        if not output_path.is_file():
            all_features = []
            for start in range(0, len(selected_paths), args.batch_size):
                batch_paths = selected_paths[start : start + args.batch_size]
                tensors = [
                    tfm(Image.open(path).convert("RGB"))
                    for path in batch_paths
                ]
                batch = torch.stack(tensors).to(device)
                with torch.no_grad():
                    features = (
                        model(pixel_values=batch)
                        .last_hidden_state[:, 0, :]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                all_features.append(features)
            features = np.concatenate(all_features, axis=0)
            if features.shape != (
                len(selected_paths),
                int(spec["feature_transport"]["required_output_dim"]),
            ):
                raise RuntimeError(
                    f"unexpected Phikon feature shape for {wsi_id}: {features.shape}"
                )

            temporary = output_path.with_suffix(".h5.tmp")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            with h5py.File(temporary, "w") as handle:
                handle.create_dataset("features", data=features, compression="gzip")
                handle.create_dataset(
                    "patch_names",
                    data=np.asarray([path.name for path in selected_paths], dtype=object),
                    dtype=string_dtype,
                )
                handle.attrs["image_id"] = wsi_id
                handle.attrs["patient_id"] = str(row.patient_id)
                handle.attrs["gleason_score"] = str(row.gleason_score)
                handle.attrs["isup_grade"] = int(row.isup_grade)
                handle.attrs["phikon_snapshot_revision"] = str(
                    replay["snapshot_revision"]
                )
            temporary.replace(output_path)

        with h5py.File(output_path, "r") as handle:
            shape = tuple(handle["features"].shape)
            _ = handle["features"][0:1]
        if shape[1] != int(spec["feature_transport"]["required_output_dim"]):
            raise ValueError(f"invalid stored feature width for {wsi_id}: {shape}")

        manifest_rows.append(
            {
                "image_id": wsi_id,
                "patient_id": str(row.patient_id),
                "gleason_score": str(row.gleason_score),
                "isup_grade": int(row.isup_grade),
                "feature_path": str(output_path.resolve()),
                "patch_count": int(shape[0]),
                "feature_sha256": sha256(output_path),
            }
        )
        print(f"prepared {number}/{len(mapping)} {wsi_id} patches={shape[0]}", flush=True)

    manifest = pd.DataFrame(manifest_rows).sort_values("image_id").reset_index(drop=True)
    if len(manifest) != int(dataset_spec["expected_wsi_units"]):
        raise RuntimeError("external feature manifest is incomplete")
    if manifest["patient_id"].nunique() != int(dataset_spec["expected_patients"]):
        raise RuntimeError("external feature manifest patient count changed")

    manifest_path = args.output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "schema_version": "sicapv2-phikon-external-bags/v1",
        "status": "complete",
        "external_spec_sha256": sha256(args.spec),
        "replay_report_sha256": sha256(args.replay_report),
        "phikon_snapshot_revision": replay["snapshot_revision"],
        "dataset_root": str(args.dataset_root.resolve()),
        "source_patch_count": len(image_paths),
        "duplicate_patch_exclusion_count": len(excluded_duplicate_rows),
        "wsi_count": len(manifest),
        "patient_count": manifest["patient_id"].nunique(),
        "feature_dim": int(spec["feature_transport"]["required_output_dim"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
