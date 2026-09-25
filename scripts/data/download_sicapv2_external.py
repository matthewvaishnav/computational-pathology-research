#!/usr/bin/env python3
"""Download the frozen SICAPv2 v1 archive and record immutable local provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import requests


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\sicapv2_external"))
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--no-extract", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dataset = spec["external_dataset"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    archive = args.output_root / "SICAPv2_v1.zip"
    if not archive.is_file():
        url = dataset["direct_archive_url"]
        print(f"Downloading frozen SICAPv2 v1 archive to {archive}", flush=True)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            written = 0
            with archive.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=args.chunk_mb * 1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        print(f"  {written / total:.1%}", flush=True)
    archive_hash = sha256(archive)

    extract_root = args.output_root / "extracted"
    if not args.no_extract:
        extract_root.mkdir(parents=True, exist_ok=True)
        marker = extract_root / ".sicapv2_v1_extracted"
        if not marker.exists():
            with zipfile.ZipFile(archive, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise RuntimeError(f"ZIP CRC failure: {bad}")
                zf.extractall(extract_root)
            marker.write_text(archive_hash + "\n", encoding="utf-8")
        elif marker.read_text(encoding="utf-8").strip() != archive_hash:
            raise RuntimeError("existing extracted tree belongs to a different archive hash")

    payload = {
        "schema_version": "sicapv2-v1-download/v1",
        "dataset": dataset["name"],
        "version": dataset["version"],
        "doi": dataset["doi"],
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "extract_root": str(extract_root),
        "extracted": not args.no_extract,
    }
    out = args.output_root / "download_manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
