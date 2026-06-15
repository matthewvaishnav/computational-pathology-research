#!/usr/bin/env python3
"""Download and verify the public SCORPION dataset release from Zenodo.

Official release
----------------
Record: https://zenodo.org/records/16517924
DOI: 10.5281/zenodo.16517924
License: CC BY 4.0
Archive: SCORPION_dataset.zip
MD5: 21e4c586c76a42f3a69b6a5361882a01

The script uses only the Python standard library. It downloads to a temporary
``.part`` file, verifies the published MD5 checksum, atomically renames the
archive, optionally extracts it with path-traversal protection, and writes a
small provenance JSON record.

Example
-------
python scripts/scorpion/download_scorpion_dataset.py \
    --output-dir data/scorpion \
    --extract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ZENODO_RECORD = "https://zenodo.org/records/16517924"
DOI = "10.5281/zenodo.16517924"
LICENSE = "CC BY 4.0"
ARCHIVE_NAME = "SCORPION_dataset.zip"
DOWNLOAD_URL = (
    "https://zenodo.org/records/16517924/files/"
    "SCORPION_dataset.zip?download=1"
)
EXPECTED_MD5 = "21e4c586c76a42f3a69b6a5361882a01"
CHUNK_SIZE = 8 * 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when download, verification, or extraction fails."""


def md5_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return a streaming MD5 checksum for a file."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def human_bytes(value: int) -> str:
    """Format a byte count for console progress."""
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def download_file(url: str, destination: Path, force: bool = False) -> None:
    """Stream a URL to destination via a temporary file.

    Existing verified archives are reused. Partial files are restarted rather
    than silently concatenated because public object stores do not always honor
    HTTP range requests consistently.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and not force:
        observed = md5_file(destination)
        if observed == EXPECTED_MD5:
            print(f"Verified archive already exists: {destination}")
            return
        raise DownloadError(
            f"Existing archive checksum mismatch: {destination}\n"
            f"expected={EXPECTED_MD5} observed={observed}\n"
            "Use --force to replace it."
        )

    part_path = destination.with_suffix(destination.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "computational-pathology-research/SCORPION-downloader",
            "Accept": "application/octet-stream",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = _content_length(response)
            downloaded = 0
            with part_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = 100.0 * downloaded / total
                        message = (
                            f"\rDownloaded {human_bytes(downloaded)} / "
                            f"{human_bytes(total)} ({percent:5.1f}%)"
                        )
                    else:
                        message = f"\rDownloaded {human_bytes(downloaded)}"
                    print(message, end="", flush=True)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part_path.unlink(missing_ok=True)
        raise DownloadError(f"Download failed: {exc}") from exc

    print()
    if not part_path.is_file() or part_path.stat().st_size == 0:
        part_path.unlink(missing_ok=True)
        raise DownloadError("Download produced an empty file.")

    observed = md5_file(part_path)
    if observed != EXPECTED_MD5:
        bad_path = part_path.with_suffix(part_path.suffix + ".bad-md5")
        part_path.replace(bad_path)
        raise DownloadError(
            "Downloaded archive failed checksum verification.\n"
            f"expected={EXPECTED_MD5}\nobserved={observed}\n"
            f"quarantined={bad_path}"
        )

    if destination.exists():
        destination.unlink()
    part_path.replace(destination)
    print(f"Checksum verified: {observed}")
    print(f"Saved archive: {destination}")


def safe_member_destination(root: Path, member_name: str) -> Path:
    """Resolve a ZIP member and reject absolute or parent-traversal paths."""
    pure = PurePosixPath(member_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise DownloadError(f"Unsafe ZIP member path: {member_name!r}")

    destination = (root / Path(*pure.parts)).resolve()
    root_resolved = root.resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise DownloadError(f"ZIP member escapes extraction root: {member_name!r}") from exc
    return destination


def extract_archive(archive: Path, extract_dir: Path, force: bool = False) -> None:
    """Safely extract the verified archive."""
    marker = extract_dir / ".scorpion_extraction_complete.json"
    if marker.is_file() and not force:
        print(f"Extraction already marked complete: {extract_dir}")
        return

    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive) as zipped:
            bad_member = zipped.testzip()
            if bad_member is not None:
                raise DownloadError(f"ZIP integrity test failed at {bad_member!r}")

            members = zipped.infolist()
            total = len(members)
            for index, member in enumerate(members, start=1):
                target = safe_member_destination(extract_dir, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                if index == total or index % 100 == 0:
                    print(f"\rExtracted {index:,} / {total:,} entries", end="", flush=True)
            print()
    except (zipfile.BadZipFile, OSError) as exc:
        raise DownloadError(f"Extraction failed: {exc}") from exc

    marker.write_text(
        json.dumps(
            {
                "archive": str(archive.resolve()),
                "archive_md5": md5_file(archive),
                "entries": total,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Extracted dataset: {extract_dir}")


def write_provenance(output_dir: Path, archive: Path) -> Path:
    """Record the exact public release used locally."""
    path = output_dir / "scorpion_download_provenance.json"
    payload = {
        "dataset": "SCORPION",
        "record_url": ZENODO_RECORD,
        "doi": DOI,
        "license": LICENSE,
        "archive_name": ARCHIVE_NAME,
        "download_url": DOWNLOAD_URL,
        "published_md5": EXPECTED_MD5,
        "observed_md5": md5_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scorpion"),
        help="Directory for the archive and provenance record.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive after checksum verification.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Extraction directory; defaults to OUTPUT_DIR/dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing archive or extraction directory.",
    )
    args = parser.parse_args()

    archive = args.output_dir / ARCHIVE_NAME
    try:
        download_file(DOWNLOAD_URL, archive, force=args.force)
        provenance = write_provenance(args.output_dir, archive)
        print(f"Wrote provenance: {provenance}")

        if args.extract:
            extract_dir = args.extract_dir or (args.output_dir / "dataset")
            extract_archive(archive, extract_dir, force=args.force)
    except DownloadError as exc:
        print(f"SCORPION DOWNLOAD FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
