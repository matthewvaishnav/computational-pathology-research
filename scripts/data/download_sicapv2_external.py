#!/usr/bin/env python3
"""Acquire frozen SICAPv2 images and record content-level provenance.

Preferred routes are the official Mendeley v1 ZIP endpoints. If those reject
scripted access, fall back to the public Kaggle mirror and validate the actual
image corpus against the preregistered 18,783-patch / 155-WSI inventory before
any downstream feature extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)
KAGGLE_MIRROR_HANDLE = "shridharspol/sicapv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\\sicapv2_external"))
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Replace an existing local acquisition after validating no downstream result depends on it.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_id_from_doi(doi: str) -> str:
    suffix = doi.split("/", 1)[-1]
    if "." not in suffix:
        raise ValueError(f"cannot derive Mendeley dataset id from DOI: {doi}")
    return suffix.rsplit(".", 1)[0]


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36",
            "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
        }
    )
    return session


def validate_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"downloaded file is not a valid ZIP: {path}") from exc
    if bad is not None:
        raise RuntimeError(f"ZIP CRC failure: {bad}")


def find_image_root(root: Path, expected_count: int) -> Path:
    candidates: list[tuple[Path, int]] = []
    for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        if directory.name.lower() != "images":
            continue
        count = sum(1 for _ in directory.glob("*.jpg"))
        if count:
            candidates.append((directory, count))
    exact = [directory for directory, count in candidates if count == expected_count]
    if len(exact) != 1:
        raise RuntimeError(
            "expected exactly one SICAPv2 images directory with "
            f"{expected_count} JPG files; observed "
            + repr([(str(path), count) for path, count in candidates])
        )
    return exact[0]


def validate_image_corpus(
    *,
    root: Path,
    spec_path: Path,
    dataset: dict,
) -> dict[str, object]:
    expected_images = int(dataset["expected_patch_images"])
    expected_wsi = int(dataset["expected_wsi_units"])
    image_root = find_image_root(root, expected_images)
    images = sorted(image_root.glob("*.jpg"), key=lambda p: p.name)

    observed_wsi = sorted({path.stem.split("_", 1)[0] for path in images})
    mapping_path = Path(str(dataset["patient_mapping"]["local_frozen_copy"]))
    if not mapping_path.is_absolute():
        mapping_path = (Path.cwd() / mapping_path).resolve()
    mapping = pd.read_csv(mapping_path, dtype={"image_id": str, "patient_id": str})
    expected_ids = sorted(mapping["image_id"].astype(str).tolist())
    if len(expected_ids) != expected_wsi:
        raise RuntimeError(
            f"frozen mapping contains {len(expected_ids)} WSI IDs; expected {expected_wsi}"
        )
    if observed_wsi != expected_ids:
        missing = sorted(set(expected_ids) - set(observed_wsi))
        extra = sorted(set(observed_wsi) - set(expected_ids))
        raise RuntimeError(
            "SICAPv2 image corpus WSI identity mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    inventory_hasher = hashlib.sha256()
    total_bytes = 0
    for index, path in enumerate(images, 1):
        file_hash = sha256(path)
        size = path.stat().st_size
        total_bytes += size
        inventory_hasher.update(path.name.encode("utf-8"))
        inventory_hasher.update(b"\0")
        inventory_hasher.update(str(size).encode("ascii"))
        inventory_hasher.update(b"\0")
        inventory_hasher.update(file_hash.encode("ascii"))
        inventory_hasher.update(b"\n")
        if index % 1000 == 0:
            print(f"  hashed {index}/{len(images)} SICAPv2 images...", flush=True)

    return {
        "image_root": str(image_root.resolve()),
        "image_count": len(images),
        "wsi_count": len(observed_wsi),
        "image_bytes": total_bytes,
        "content_inventory_sha256": inventory_hasher.hexdigest(),
        "frozen_mapping_sha256": sha256(mapping_path),
        "spec_sha256": sha256(spec_path),
    }


def download_mendeley_archive(
    *,
    dataset: dict,
    destination: Path,
    chunk_mb: int,
) -> tuple[dict[str, str | int] | None, list[str]]:
    dataset_id = dataset_id_from_doi(str(dataset["doi"]))
    version = int(dataset["version"])
    api_zip_url = (
        f"https://api.data.mendeley.com/datasets/{dataset_id}/zip/file_downloaded"
        f"?version={version}"
    )
    legacy_url = str(dataset["direct_archive_url"])
    candidates = [
        ("mendeley_public_api_zip", api_zip_url),
        ("mendeley_legacy_public_files", legacy_url),
    ]

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    errors: list[str] = []
    session = build_session()
    try:
        for source_name, url in candidates:
            print(f"Trying {source_name}: {url}", flush=True)
            try:
                with session.get(
                    url,
                    stream=True,
                    timeout=(30, 180),
                    allow_redirects=True,
                ) as response:
                    if response.status_code >= 400:
                        errors.append(
                            f"{source_name}: HTTP {response.status_code} for {response.url}"
                        )
                        continue
                    total = int(response.headers.get("content-length", 0) or 0)
                    written = 0
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_content(
                            chunk_size=max(1, chunk_mb) * 1024 * 1024
                        ):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            written += len(chunk)
                            if total:
                                print(f"  {written / total:.1%}", flush=True)
                    validate_zip(temporary)
                    temporary.replace(destination)
                    return (
                        {
                            "download_source": source_name,
                            "download_request_url": url,
                            "download_final_host": urlparse(str(response.url)).netloc,
                            "downloaded_bytes": written,
                        },
                        errors,
                    )
            except (requests.RequestException, RuntimeError, OSError) as exc:
                errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
                if temporary.exists():
                    temporary.unlink()
    finally:
        session.close()
    return None, errors


def download_kaggle_mirror(extract_root: Path) -> dict[str, str]:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "official Mendeley routes failed and kagglehub is not installed. "
            "Install it with: python -m pip install -U kagglehub"
        ) from exc

    print(
        f"Trying Kaggle public mirror via kagglehub: {KAGGLE_MIRROR_HANDLE}",
        flush=True,
    )
    returned = kagglehub.dataset_download(
        KAGGLE_MIRROR_HANDLE,
        output_dir=str(extract_root),
    )
    return {
        "download_source": "kagglehub_public_mirror",
        "download_request_url": f"https://www.kaggle.com/datasets/{KAGGLE_MIRROR_HANDLE}",
        "download_final_host": "www.kaggle.com",
        "kagglehub_returned_path": str(returned),
    }


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dataset = spec["external_dataset"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    archive = args.output_root / "SICAPv2_v1.zip"
    extract_root = args.output_root / "extracted"

    if args.force_redownload:
        if archive.exists():
            archive.unlink()
        if extract_root.exists():
            shutil.rmtree(extract_root)

    # Preserve invalid tiny/error responses for audit, but never treat them as data.
    if archive.exists():
        try:
            validate_zip(archive)
        except RuntimeError:
            invalid_hash = sha256(archive)
            invalid_path = archive.with_name(
                f"{archive.stem}.invalid-{invalid_hash[:12]}{archive.suffix}.bin"
            )
            archive.replace(invalid_path)
            print(f"Quarantined invalid local archive as {invalid_path}", flush=True)

    download_meta: dict[str, object]
    route_errors: list[str] = []

    if archive.is_file():
        validate_zip(archive)
        download_meta = {
            "download_source": "existing_local_archive",
            "download_request_url": "",
            "download_final_host": "",
            "downloaded_bytes": archive.stat().st_size,
        }
        if not args.no_extract:
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(extract_root)
    else:
        print(f"Acquiring frozen SICAPv2 v1 into {args.output_root}", flush=True)
        meta, route_errors = download_mendeley_archive(
            dataset=dataset,
            destination=archive,
            chunk_mb=args.chunk_mb,
        )
        if meta is not None:
            download_meta = dict(meta)
            if not args.no_extract:
                extract_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(extract_root)
        else:
            if args.no_extract:
                raise RuntimeError(
                    "Mendeley routes failed and --no-extract prevents directory-mirror fallback:\n  - "
                    + "\n  - ".join(route_errors)
                )
            extract_root.mkdir(parents=True, exist_ok=True)
            download_meta = download_kaggle_mirror(extract_root)

    if args.no_extract:
        if not archive.is_file():
            raise RuntimeError("no valid SICAPv2 archive was acquired")
        payload = {
            "schema_version": "sicapv2-v1-download/v2",
            "dataset": dataset["name"],
            "version": dataset["version"],
            "doi": dataset["doi"],
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive),
            "extracted": False,
            "route_errors": route_errors,
            **download_meta,
        }
    else:
        corpus = validate_image_corpus(
            root=extract_root,
            spec_path=args.spec,
            dataset=dataset,
        )
        payload = {
            "schema_version": "sicapv2-v1-download/v2",
            "dataset": dataset["name"],
            "version": dataset["version"],
            "doi": dataset["doi"],
            "archive_path": str(archive) if archive.is_file() else None,
            "archive_size_bytes": archive.stat().st_size if archive.is_file() else None,
            "archive_sha256": sha256(archive) if archive.is_file() else None,
            "extract_root": str(extract_root.resolve()),
            "extracted": True,
            "route_errors": route_errors,
            **download_meta,
            **corpus,
        }

    out = args.output_root / "download_manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
