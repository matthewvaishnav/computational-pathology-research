"""Fail-closed Hugging Face release preparation and publishing.

GitHub remains the authoritative engineering and evidence record. This module
publishes only explicitly described release folders and records the immutable
Hub revision back into the program release registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

ALLOWED_RELEASE_STATES = {"released", "private", "prepared", "deferred", "withdrawn"}
ALLOWED_VISIBILITIES = {"public", "private", "none"}
ALLOWED_REPO_TYPES = {"dataset", "model", "space"}
REQUIRED_REGISTRY_FIELDS = {
    "id",
    "research_line",
    "artifact_type",
    "hf_repo",
    "visibility",
    "release_state",
    "github_repo",
    "github_commit",
    "source_artifacts",
    "evidence_status",
    "claim_boundary",
    "license",
    "checksums",
    "created_at",
    "last_verified_at",
}
DATASET_CARD_SECTIONS = {
    "data origin",
    "unit of observation",
    "schema",
    "sample counts",
    "preprocessing",
    "coordinates",
    "exclusions",
    "checksums",
    "licensing",
    "intended use",
    "limitations",
    "provenance",
    "citation",
    "claim boundary",
}
MODEL_CARD_SECTIONS = {
    "method summary",
    "architecture",
    "intended use",
    "out-of-scope use",
    "training data",
    "evaluation",
    "supported claims",
    "unsupported claims",
    "limitations",
    "provenance",
    "citation",
    "license",
    "reproducibility",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class ReleaseError(RuntimeError):
    """Raised when a release gate fails closed."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseError(f"Could not read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"Expected a YAML mapping in {path}")
    return value


def write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    temporary.replace(path)


def _require_nonempty_text(
    record: Mapping[str, Any], field: str, record_id: str
) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseError(
            f"Registry record {record_id!r} requires non-empty {field!r}"
        )


def validate_registry_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema_version") != "huggingface-release-registry/v1":
        raise ReleaseError("Unsupported or missing release-registry schema_version")
    records = document.get("releases")
    if not isinstance(records, list):
        raise ReleaseError("Registry 'releases' must be a list")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ReleaseError(f"Registry item {index} must be a mapping")
        missing = REQUIRED_REGISTRY_FIELDS - raw_record.keys()
        if missing:
            raise ReleaseError(f"Registry item {index} is missing: {sorted(missing)}")
        record = dict(raw_record)
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ReleaseError(f"Registry item {index} has an invalid id")
        if record_id in seen:
            raise ReleaseError(f"Duplicate registry id: {record_id}")
        seen.add(record_id)

        if record["release_state"] not in ALLOWED_RELEASE_STATES:
            raise ReleaseError(
                f"Registry record {record_id!r} has invalid release_state {record['release_state']!r}"
            )
        if record["visibility"] not in ALLOWED_VISIBILITIES:
            raise ReleaseError(
                f"Registry record {record_id!r} has invalid visibility {record['visibility']!r}"
            )
        if record["artifact_type"] not in ALLOWED_REPO_TYPES:
            raise ReleaseError(
                f"Registry record {record_id!r} has invalid artifact_type "
                f"{record['artifact_type']!r}"
            )
        if record["visibility"] == "none" and record["hf_repo"] is not None:
            raise ReleaseError(
                f"Registry record {record_id!r} must not name an HF repo"
            )
        if record["visibility"] != "none":
            _require_nonempty_text(record, "hf_repo", record_id)
        for text_field in (
            "research_line",
            "github_repo",
            "evidence_status",
            "claim_boundary",
            "license",
        ):
            _require_nonempty_text(record, text_field, record_id)
        commit = record["github_commit"]
        if commit is not None and (
            not isinstance(commit, str) or not GIT_SHA_RE.fullmatch(commit)
        ):
            raise ReleaseError(
                f"Registry record {record_id!r} has an invalid github_commit"
            )
        if not isinstance(record["source_artifacts"], list):
            raise ReleaseError(
                f"Registry record {record_id!r} source_artifacts must be a list"
            )
        checksums = record["checksums"]
        if not isinstance(checksums, dict):
            raise ReleaseError(
                f"Registry record {record_id!r} checksums must be a mapping"
            )
        for name, digest in checksums.items():
            if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or not SHA256_RE.fullmatch(digest)
            ):
                raise ReleaseError(
                    f"Registry record {record_id!r} has an invalid checksum"
                )
        validated.append(record)
    return validated


def validate_registry(path: Path) -> list[dict[str, Any]]:
    return validate_registry_document(load_yaml(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_release_files(folder: Path) -> Iterable[Path]:
    root = folder.resolve()
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ReleaseError(f"Symlinks are prohibited in release folders: {path}")
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ReleaseError(f"Release file escapes folder root: {path}")
        yield path


def compute_checksums(folder: Path) -> dict[str, str]:
    if not folder.is_dir():
        raise ReleaseError(f"Release folder does not exist: {folder}")
    return {
        path.relative_to(folder).as_posix(): sha256_file(path)
        for path in _safe_release_files(folder)
    }


def write_checksum_manifest(folder: Path) -> dict[str, str]:
    checksums = compute_checksums(folder)
    if not checksums:
        raise ReleaseError("Refusing to write an empty checksum manifest")
    body = "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    (folder / "checksums.sha256").write_text(body, encoding="utf-8", newline="\n")
    return checksums


def read_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ReleaseError(
                f"Malformed checksum line {line_number} in {path}"
            ) from exc
        if not SHA256_RE.fullmatch(digest) or not name or name in checksums:
            raise ReleaseError(f"Invalid checksum line {line_number} in {path}")
        checksums[name] = digest
    if not checksums:
        raise ReleaseError(f"Checksum manifest is empty: {path}")
    return checksums


def verify_checksum_manifest(folder: Path) -> dict[str, str]:
    manifest_path = folder / "checksums.sha256"
    if not manifest_path.is_file():
        raise ReleaseError(f"Missing checksum manifest: {manifest_path}")
    expected = read_checksum_manifest(manifest_path)
    observed = compute_checksums(folder)
    if set(expected) != set(observed):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ReleaseError(
            f"Checksum inventory mismatch; missing={missing}, extra={extra}"
        )
    mismatched = sorted(name for name in expected if expected[name] != observed[name])
    if mismatched:
        raise ReleaseError(f"SHA256 mismatch for: {mismatched}")
    return observed


def _frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ReleaseError("Card must begin with YAML front matter")
    try:
        closing = next(
            i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ReleaseError("Card YAML front matter is not closed") from exc
    try:
        metadata = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise ReleaseError(f"Invalid card YAML front matter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ReleaseError("Card YAML front matter must be a mapping")
    if not metadata.get("license"):
        raise ReleaseError("Card YAML front matter requires a license")
    return metadata


def validate_card(path: Path, artifact_type: str) -> None:
    if artifact_type not in {"dataset", "model"}:
        raise ReleaseError("Card validation supports dataset or model artifacts")
    text = path.read_text(encoding="utf-8")
    _frontmatter(text)
    headings = {
        match.group(1).strip().lower()
        for match in re.finditer(r"^#{2,4}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    }
    required = (
        DATASET_CARD_SECTIONS if artifact_type == "dataset" else MODEL_CARD_SECTIONS
    )
    missing = sorted(required - headings)
    if missing:
        raise ReleaseError(f"Card {path} is missing required sections: {missing}")


def validate_release_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "id",
        "repo_id",
        "repo_type",
        "visibility",
        "card",
        "source_paths",
        "github_repo",
        "github_commit",
    }
    missing = required - spec.keys()
    if missing:
        raise ReleaseError(f"Release spec is missing: {sorted(missing)}")
    if spec["schema_version"] != "huggingface-release-spec/v1":
        raise ReleaseError("Unsupported release spec schema_version")
    if spec["repo_type"] not in ALLOWED_REPO_TYPES:
        raise ReleaseError(f"Unsupported repo_type: {spec['repo_type']}")
    if spec["visibility"] not in {"public", "private"}:
        raise ReleaseError("Release visibility must be public or private")
    if not isinstance(spec["source_paths"], list):
        raise ReleaseError("Release source_paths must be a list")
    if not GIT_SHA_RE.fullmatch(str(spec["github_commit"])):
        raise ReleaseError("Release spec requires an immutable Git commit SHA")
    return dict(spec)


def _relative_source(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ReleaseError(
            f"Release source path must be repository-relative: {raw_path}"
        )
    root_resolved = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ReleaseError(f"Release source path escapes repository root: {raw_path}")
    return resolved


def prepare_release(
    spec_path: Path,
    repository_root: Path,
    output_folder: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    spec = validate_release_spec(load_yaml(spec_path))
    if spec.get("blocked"):
        raise ReleaseError(
            f"Release spec is blocked: {spec.get('blocker', 'unspecified blocker')}"
        )
    if output_folder.exists():
        raise ReleaseError(f"Output folder already exists: {output_folder}")
    card = _relative_source(repository_root, str(spec["card"]))
    validate_card(card, str(spec["repo_type"]))
    sources = [
        _relative_source(repository_root, str(item)) for item in spec["source_paths"]
    ]
    if not sources:
        raise ReleaseError("Release spec has no source artifacts")
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise ReleaseError(f"Release source artifacts are missing: {missing}")

    plan = {
        "id": spec["id"],
        "repo_id": spec["repo_id"],
        "visibility": spec["visibility"],
        "sources": [
            str(path.relative_to(repository_root.resolve())) for path in sources
        ],
        "output_folder": str(output_folder),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    output_folder.mkdir(parents=True)
    try:
        shutil.copy2(card, output_folder / "README.md")
        for source in sources:
            destination = output_folder / source.relative_to(repository_root.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
            else:
                raise ReleaseError(f"Unsupported source artifact: {source}")
        provenance = {
            "schema_version": "huggingface-release-provenance/v1",
            "release_id": spec["id"],
            "repo_id": spec["repo_id"],
            "github_repo": spec["github_repo"],
            "github_commit": spec["github_commit"],
            "source_paths": spec["source_paths"],
            "prepared_at": utc_now(),
            "release_spec_sha256": sha256_file(spec_path),
        }
        (output_folder / "release-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        plan["checksums"] = write_checksum_manifest(output_folder)
        verify_checksum_manifest(output_folder)
    except Exception:
        shutil.rmtree(output_folder, ignore_errors=True)
        raise
    return plan


def _load_hf() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, get_token, hf_hub_download
    except ImportError as exc:
        raise ReleaseError(
            "huggingface_hub is required: python -m pip install -r "
            "tools/huggingface/requirements.txt"
        ) from exc
    return HfApi, get_token, hf_hub_download


def authentication_status() -> dict[str, Any]:
    HfApi, get_token, _ = _load_hf()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        return {
            "authenticated": False,
            "write_capable": False,
            "name": None,
            "role": None,
        }
    info = HfApi(token=token).whoami(token=token)
    role = info.get("auth", {}).get("accessToken", {}).get("role")
    return {
        "authenticated": True,
        "write_capable": role in {"write", "fineGrained"},
        "name": info.get("name") or info.get("fullname"),
        "role": role or "unknown",
    }


def verify_remote_release(
    repo_id: str,
    repo_type: str,
    folder: Path,
    *,
    revision: str = "main",
    token: Optional[str] = None,
) -> dict[str, str]:
    HfApi, get_token, hf_hub_download = _load_hf()
    token = token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise ReleaseError("No Hugging Face credential is available")
    local_checksums = verify_checksum_manifest(folder)
    api = HfApi(token=token)
    remote_files = set(
        api.list_repo_files(
            repo_id, repo_type=repo_type, revision=revision, token=token
        )
    )
    required_files = set(local_checksums) | {"checksums.sha256"}
    missing = sorted(required_files - remote_files)
    if missing:
        raise ReleaseError(f"Remote release is missing files: {missing}")
    with tempfile.TemporaryDirectory(prefix="hf-release-verify-") as temporary:
        for relative_path, expected in local_checksums.items():
            downloaded = Path(
                hf_hub_download(
                    repo_id,
                    filename=relative_path,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                    cache_dir=temporary,
                )
            )
            observed = sha256_file(downloaded)
            if observed != expected:
                raise ReleaseError(f"Remote SHA256 mismatch for {relative_path}")
    return local_checksums


def update_registry_after_publish(
    registry_path: Path,
    release_id: str,
    *,
    revision: str,
    checksums: Mapping[str, str],
    visibility: str,
) -> None:
    document = load_yaml(registry_path)
    records = validate_registry_document(document)
    matches = [record for record in records if record["id"] == release_id]
    if len(matches) != 1:
        raise ReleaseError(f"Expected exactly one registry record for {release_id!r}")
    record = matches[0]
    record["release_state"] = "private" if visibility == "private" else "released"
    record["hf_revision"] = revision
    record["checksums"] = dict(sorted(checksums.items()))
    record["created_at"] = record.get("created_at") or utc_now()
    record["last_verified_at"] = utc_now()
    document["releases"] = records
    write_yaml_atomic(registry_path, document)


def publish_release(
    spec_path: Path,
    folder: Path,
    registry_path: Path,
    *,
    dry_run: bool = False,
    allow_public: bool = False,
) -> dict[str, Any]:
    spec = validate_release_spec(load_yaml(spec_path))
    if spec.get("blocked"):
        raise ReleaseError(
            f"Release spec is blocked: {spec.get('blocker', 'unspecified blocker')}"
        )
    if spec["visibility"] == "public" and not allow_public:
        raise ReleaseError(
            "Public publishing requires the explicit --allow-public flag"
        )
    validate_card(folder / "README.md", str(spec["repo_type"]))
    checksums = verify_checksum_manifest(folder)
    plan = {
        "repo_id": spec["repo_id"],
        "repo_type": spec["repo_type"],
        "visibility": spec["visibility"],
        "files": sorted(checksums),
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    HfApi, get_token, _ = _load_hf()
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise ReleaseError("No Hugging Face credential is available")
    status = authentication_status()
    if not status["write_capable"]:
        raise ReleaseError(
            f"Hugging Face credential is not write-capable (role={status['role']})"
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=str(spec["repo_id"]),
        repo_type=str(spec["repo_type"]),
        private=spec["visibility"] == "private",
        exist_ok=True,
        token=token,
    )
    commit = api.upload_folder(
        repo_id=str(spec["repo_id"]),
        repo_type=str(spec["repo_type"]),
        folder_path=str(folder),
        path_in_repo="",
        commit_message=str(spec.get("commit_message") or f"Release {spec['id']}"),
        token=token,
    )
    revision = str(
        getattr(commit, "oid", None) or getattr(commit, "commit_id", None) or "main"
    )
    verified = verify_remote_release(
        str(spec["repo_id"]),
        str(spec["repo_type"]),
        folder,
        revision=revision,
        token=token,
    )
    update_registry_after_publish(
        registry_path,
        str(spec["id"]),
        revision=revision,
        checksums=verified,
        visibility=str(spec["visibility"]),
    )
    plan.update({"revision": revision, "verified": True, "dry_run": False})
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth = subparsers.add_parser(
        "auth-status", help="Report authentication without exposing tokens"
    )
    auth.set_defaults(handler=lambda args: authentication_status())

    registry = subparsers.add_parser(
        "validate-registry", help="Validate the release registry"
    )
    registry.add_argument("registry", type=Path)
    registry.set_defaults(
        handler=lambda args: {"records": len(validate_registry(args.registry))}
    )

    card = subparsers.add_parser(
        "validate-card", help="Validate a dataset or model card"
    )
    card.add_argument("card", type=Path)
    card.add_argument("--type", choices=["dataset", "model"], required=True)
    card.set_defaults(
        handler=lambda args: validate_card(args.card, args.type) or {"valid": True}
    )

    checksums = subparsers.add_parser(
        "checksums", help="Write and verify checksums.sha256"
    )
    checksums.add_argument("folder", type=Path)
    checksums.set_defaults(handler=lambda args: write_checksum_manifest(args.folder))

    verify = subparsers.add_parser(
        "verify-local", help="Verify a local checksum manifest"
    )
    verify.add_argument("folder", type=Path)
    verify.set_defaults(handler=lambda args: verify_checksum_manifest(args.folder))

    prepare = subparsers.add_parser(
        "prepare", help="Build a release folder from tracked sources"
    )
    prepare.add_argument("spec", type=Path)
    prepare.add_argument("--repository-root", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(
        handler=lambda args: prepare_release(
            args.spec, args.repository_root, args.output, dry_run=args.dry_run
        )
    )

    publish = subparsers.add_parser(
        "publish", help="Create/upload/verify a Hub release"
    )
    publish.add_argument("spec", type=Path)
    publish.add_argument("folder", type=Path)
    publish.add_argument("--registry", type=Path, required=True)
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--allow-public", action="store_true")
    publish.set_defaults(
        handler=lambda args: publish_release(
            args.spec,
            args.folder,
            args.registry,
            dry_run=args.dry_run,
            allow_public=args.allow_public,
        )
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (ReleaseError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
