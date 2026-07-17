#!/usr/bin/env python3
"""Fetch and normalize public Figshare metadata for the PLISM feasibility study.

This script intentionally downloads metadata only. It never downloads image payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE_DIR / "figshare_manifest.json"
DEFAULT_STORAGE_PLAN = PACKAGE_DIR / "storage_plan.md"
API_ROOT = "https://api.figshare.com/v2"
DEPOSITS = {
    "plism_registered_tiles": 23614422,
    "plism_original_wsi": 24988074,
}
SCHEMA_VERSION = "plism_figshare_manifest_v1"


class ManifestError(Exception):
    """Fail-closed metadata or manifest validation error."""


def stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_json(url: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "computational-pathology-research-plism-manifest/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ManifestError(f"unexpected_http_status:{response.status}:{url}")
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ManifestError(f"metadata_fetch_failed:{url}:{exc}") from exc


def require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        raise ManifestError(f"missing_or_invalid:{key}")
    return value


def normalize_file(file_record: dict[str, Any]) -> dict[str, Any]:
    file_id = require(file_record, "id", int)
    name = require(file_record, "name", str).strip()
    size = require(file_record, "size", int)
    download_url = require(file_record, "download_url", str).strip()
    if not name or size < 0 or not download_url.startswith("https://"):
        raise ManifestError(f"invalid_file_record:{file_id}")

    supplied_md5 = file_record.get("supplied_md5") or file_record.get("computed_md5")
    if supplied_md5 is not None:
        if not isinstance(supplied_md5, str) or len(supplied_md5) != 32:
            raise ManifestError(f"invalid_md5:{file_id}")
        supplied_md5 = supplied_md5.lower()

    return {
        "file_id": file_id,
        "name": name,
        "size_bytes": size,
        "download_url": download_url,
        "figshare_md5": supplied_md5,
    }


def normalize_article(dataset_id: str, article: dict[str, Any]) -> dict[str, Any]:
    article_id = require(article, "id", int)
    title = require(article, "title", str).strip()
    doi = require(article, "doi", str).strip()
    url = require(article, "url_public_api", str).strip()
    files_raw = require(article, "files", list)
    license_record = require(article, "license", dict)
    license_name = require(license_record, "name", str).strip()

    if article_id != DEPOSITS[dataset_id]:
        raise ManifestError(f"article_id_mismatch:{dataset_id}:{article_id}")
    if not title or not doi or not url.startswith("https://"):
        raise ManifestError(f"invalid_article_identity:{dataset_id}")
    if not files_raw:
        raise ManifestError(f"article_has_no_files:{dataset_id}")

    files = sorted((normalize_file(item) for item in files_raw), key=lambda item: item["file_id"])
    file_ids = [item["file_id"] for item in files]
    if len(file_ids) != len(set(file_ids)):
        raise ManifestError(f"duplicate_file_id:{dataset_id}")

    total_size = sum(item["size_bytes"] for item in files)
    return {
        "dataset_id": dataset_id,
        "article_id": article_id,
        "title": title,
        "doi": doi,
        "url_public_api": url,
        "figshare_url": article.get("figshare_url"),
        "version": article.get("version"),
        "published_date": article.get("published_date"),
        "modified_date": article.get("modified_date"),
        "license": license_name,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": files,
    }


def build_manifest(fetcher=fetch_json) -> dict[str, Any]:
    deposits = []
    for dataset_id, article_id in sorted(DEPOSITS.items()):
        article = fetcher(f"{API_ROOT}/articles/{article_id}")
        if not isinstance(article, dict):
            raise ManifestError(f"article_response_not_object:{dataset_id}")
        deposits.append(normalize_article(dataset_id, article))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_api": API_ROOT,
        "metadata_only": True,
        "claim_boundaries": [
            "serial sections are not identical tissue or pixel counterfactuals",
            "scanner model labels are not verified physical device identities",
            "registered coordinates are correspondences rather than causal provenance",
            "missing batch order and source-event metadata remain missing",
        ],
        "deposits": deposits,
    }
    payload["total_size_bytes"] = sum(item["total_size_bytes"] for item in deposits)
    payload["manifest_fingerprint"] = sha256_text(stable_json(payload))
    return payload


def validate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("invalid_schema_version")
    if payload.get("metadata_only") is not True:
        raise ManifestError("metadata_only_flag_missing")
    deposits = payload.get("deposits")
    if not isinstance(deposits, list) or len(deposits) != len(DEPOSITS):
        raise ManifestError("invalid_deposit_count")
    expected_ids = set(DEPOSITS)
    actual_ids = {item.get("dataset_id") for item in deposits if isinstance(item, dict)}
    if actual_ids != expected_ids:
        raise ManifestError("deposit_identity_mismatch")
    stored_fingerprint = payload.get("manifest_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("manifest_fingerprint", None)
    expected_fingerprint = sha256_text(stable_json(without_fingerprint))
    if stored_fingerprint != expected_fingerprint:
        raise ManifestError("manifest_fingerprint_mismatch")


def gibibytes(size_bytes: int) -> float:
    return size_bytes / (1024 ** 3)


def render_storage_plan(manifest: dict[str, Any]) -> str:
    lines = [
        "# PLISM metadata-derived storage plan",
        "",
        f"Manifest fingerprint: `{manifest['manifest_fingerprint']}`",
        "",
        "This plan is generated from public Figshare file metadata. It does not download image payloads.",
        "",
        "## Deposit totals",
        "",
        "| Deposit | Files | Bytes | GiB |",
        "|---|---:|---:|---:|",
    ]
    for deposit in manifest["deposits"]:
        lines.append(
            f"| `{deposit['dataset_id']}` | {deposit['file_count']} | "
            f"{deposit['total_size_bytes']} | {gibibytes(deposit['total_size_bytes']):.2f} |"
        )
    lines.extend([
        "",
        f"Combined advertised payload: **{gibibytes(manifest['total_size_bytes']):.2f} GiB**.",
        "",
        "## Staged acquisition decision",
        "",
        "1. Begin with the registered-tile deposit for loader, grouping, and paired-sampling validation.",
        "2. Select a small tissue × stain × scanner subset only after filenames and manifest fields pass contradiction checks.",
        "3. Download original WSIs only for groups needed to test whether tile-level findings survive WSI-derived sampling.",
        "4. Verify every downloaded file against the checksum supplied by Figshare when one is present.",
        "5. Do not mirror the full archive onto a nearly full system drive without a separate storage decision.",
        "",
        "## Scientific boundary",
        "",
        "Storage feasibility does not repair missing preparation batch, scan batch, acquisition order, exact section distance, physical device identity, or immutable acquisition-event provenance.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def fixture_article(article_id: int, name: str) -> dict[str, Any]:
    return {
        "id": article_id,
        "title": name,
        "doi": f"10.0000/{article_id}",
        "url_public_api": f"https://api.figshare.com/v2/articles/{article_id}",
        "figshare_url": f"https://figshare.com/articles/dataset/{article_id}",
        "version": 1,
        "published_date": "2024-01-01T00:00:00Z",
        "modified_date": "2024-01-01T00:00:00Z",
        "license": {"name": "CC BY 4.0"},
        "files": [
            {
                "id": article_id * 10,
                "name": "fixture.zip",
                "size": 1024,
                "download_url": "https://example.invalid/fixture.zip",
                "supplied_md5": "0" * 32,
            }
        ],
    }


def run_self_tests() -> None:
    fixtures = {
        article_id: fixture_article(article_id, dataset_id)
        for dataset_id, article_id in DEPOSITS.items()
    }

    def fake_fetch(url: str) -> dict[str, Any]:
        article_id = int(url.rsplit("/", 1)[-1])
        return fixtures[article_id]

    manifest = build_manifest(fetcher=fake_fetch)
    validate_manifest(manifest)
    assert manifest["total_size_bytes"] == 2048
    assert "registered-tile" in render_storage_plan(manifest)

    broken = json.loads(stable_json(manifest))
    broken["total_size_bytes"] += 1
    try:
        validate_manifest(broken)
    except ManifestError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("tampered manifest should fail validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--storage-plan", type=Path, default=DEFAULT_STORAGE_PLAN)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_self_tests()
            print("self-tests: passed")
            return 0
        if args.check_manifest:
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            validate_manifest(payload)
            print(f"manifest valid: {payload['manifest_fingerprint']}")
            return 0

        manifest = build_manifest()
        validate_manifest(manifest)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(stable_json(manifest), encoding="utf-8", newline="\n")
        args.storage_plan.write_text(render_storage_plan(manifest), encoding="utf-8", newline="\n")
        print(f"wrote metadata manifest: {args.manifest}")
        print(f"wrote storage plan: {args.storage_plan}")
        print(f"advertised payload GiB: {gibibytes(manifest['total_size_bytes']):.2f}")
        return 0
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
