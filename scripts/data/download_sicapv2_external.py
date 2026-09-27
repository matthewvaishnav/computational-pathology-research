#!/usr/bin/env python3
"""Download the frozen SICAPv2 v1 archive and record immutable local provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\sicapv2_external"))
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Replace an existing local archive after validating no downstream result depends on it.",
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


def download_archive(
    *,
    dataset: dict,
    destination: Path,
    chunk_mb: int,
) -> dict[str, str | int]:
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
                    return {
                        "download_source": source_name,
                        "download_request_url": url,
                        "download_final_host": urlparse(str(response.url)).netloc,
                        "downloaded_bytes": written,
                    }
            except (requests.RequestException, RuntimeError, OSError) as exc:
                errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
                if temporary.exists():
                    temporary.unlink()
    finally:
        session.close()

    raise RuntimeError(
        "all frozen SICAPv2 download routes failed:\n  - " + "\n  - ".join(errors)
    )


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dataset = spec["external_dataset"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    archive = args.output_root / "SICAPv2_v1.zip"
    download_meta: dict[str, str | int] = {
        "download_source": "existing_local_archive",
        "download_request_url": "",
        "download_final_host": "",
        "downloaded_bytes": archive.stat().st_size if archive.exists() else 0,
    }

    if args.force_redownload and archive.exists():
        archive.unlink()

    if not archive.is_file():
        print(f"Downloading frozen SICAPv2 v1 archive to {archive}", flush=True)
        download_meta = download_archive(
            dataset=dataset,
            destination=archive,
            chunk_mb=args.chunk_mb,
        )

    validate_zip(archive)
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
        **download_meta,
    }
    out = args.output_root / "download_manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
