#!/usr/bin/env python3
"""List or download the official Multi-Scanner SCC Zenodo record.

The record identifier is frozen to the dataset DOI used by the publication:
10.5281/zenodo.7418555. Files are downloaded outside the repository and are
verified against the checksums returned by Zenodo.

Examples:
    python scripts/pathoalign_real_pairs/download_multiscanner_scc.py

    python scripts/pathoalign_real_pairs/download_multiscanner_scc.py \
        --download --output-dir D:/datasets/multiscanner_scc
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import requests


RECORD_ID = 7418555
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "datasets" / "multiscanner_scc",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files after listing them. Without this flag, no data are written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def fetch_record(timeout: int) -> dict:
    response = requests.get(API_URL, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("id", -1)) != RECORD_ID:
        raise RuntimeError(
            f"Zenodo returned record {payload.get('id')} instead of {RECORD_ID}."
        )
    return payload


def checksum_parts(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Unsupported Zenodo checksum format: {value}")
    algorithm, digest = value.split(":", 1)
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    return algorithm, digest.lower()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def iter_files(record: dict) -> Iterator[dict]:
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("The Zenodo record does not contain downloadable files.")
    for item in sorted(files, key=lambda value: str(value.get("key", ""))):
        required = ("key", "size", "checksum", "links")
        missing = [name for name in required if name not in item]
        if missing:
            raise RuntimeError(f"Zenodo file entry is missing {missing}: {item}")
        yield item


def download_file(item: dict, output_dir: Path, timeout: int) -> Path:
    key = str(item["key"])
    expected_size = int(item["size"])
    algorithm, expected_digest = checksum_parts(str(item["checksum"]))
    links = item["links"]
    url = links.get("content") or links.get("self")
    if not url:
        raise RuntimeError(f"No content URL for Zenodo file: {key}")

    destination = output_dir / key
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size == expected_size:
        actual_digest = file_digest(destination, algorithm)
        if actual_digest == expected_digest:
            print(f"Verified existing: {destination}")
            return destination
        print(f"Checksum mismatch; replacing: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    print(f"Downloading {key} ({human_bytes(expected_size)})")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {key}: expected {expected_size}, got {actual_size}."
        )
    actual_digest = file_digest(partial, algorithm)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Checksum mismatch for {key}: expected {expected_digest}, got {actual_digest}."
        )

    partial.replace(destination)
    print(f"Verified: {destination}")
    return destination


def main() -> None:
    args = parse_args()
    record = fetch_record(args.timeout)
    files = list(iter_files(record))
    total_size = sum(int(item["size"]) for item in files)

    metadata = record.get("metadata", {})
    print(f"Title: {metadata.get('title', 'unknown')}")
    print(f"Zenodo record: {RECORD_ID}")
    print(f"Files: {len(files)}")
    print(f"Total size: {human_bytes(total_size)}")
    print()
    for item in files:
        print(
            f"- {item['key']} | {human_bytes(int(item['size']))} | "
            f"{item['checksum']}"
        )

    if not args.download:
        print("\nListing only. Re-run with --download to fetch the dataset.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "zenodo_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    for item in files:
        download_file(item, args.output_dir, args.timeout)
    print(f"\nDataset downloaded and verified at: {args.output_dir}")


if __name__ == "__main__":
    main()
