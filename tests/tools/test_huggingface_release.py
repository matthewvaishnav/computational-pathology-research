import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).parents[2] / "tools" / "huggingface" / "release.py"
SPEC = importlib.util.spec_from_file_location("huggingface_release", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def dataset_card() -> str:
    headings = sorted(release.DATASET_CARD_SECTIONS)
    body = "\n\n".join(
        f"## {heading.title()}\n\nExact bounded content." for heading in headings
    )
    return f"---\nlicense: mit\n---\n\n# Evidence\n\n{body}\n"


def registry_record() -> dict:
    return {
        "id": "artifact-v1",
        "research_line": "bounded research line",
        "artifact_type": "dataset",
        "hf_repo": "MatthewVaishnav/artifact-v1",
        "visibility": "private",
        "release_state": "prepared",
        "github_repo": "matthewvaishnav/computational-pathology-research",
        "github_commit": "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce",
        "source_artifacts": ["evidence/example.json"],
        "evidence_status": "validated engineering artifact",
        "claim_boundary": "does not establish clinical utility",
        "license": "MIT",
        "checksums": {},
        "created_at": None,
        "last_verified_at": "2026-08-12T16:30:00Z",
    }


def test_tracked_registry_and_cards_validate():
    root = Path(__file__).parents[2]
    records = release.validate_registry(
        root / "docs/releases/huggingface-release-registry.yaml"
    )
    assert {record["id"] for record in records} >= {
        "panf-evidence-20260726-v1",
        "panda-phikon-wsi-spatial-features-300-v1",
        "wsi-nca-phase-a",
        "transnnmil-repaired-canonical",
        "pathologyfl-fair-weights-h",
    }
    release.validate_card(
        root
        / "docs/releases/huggingface/paired-acquisition-factorization-evidence/README.md",
        "dataset",
    )
    release.validate_card(
        root / "docs/releases/huggingface/panda-phikon-wsi-spatial-features/README.md",
        "dataset",
    )


def test_registry_rejects_duplicate_ids():
    record = registry_record()
    document = {
        "schema_version": "huggingface-release-registry/v1",
        "releases": [record, copy.deepcopy(record)],
    }
    with pytest.raises(release.ReleaseError, match="Duplicate registry id"):
        release.validate_registry_document(document)


def test_registry_rejects_invalid_checksum():
    record = registry_record()
    record["checksums"] = {"file.json": "not-a-sha256"}
    document = {
        "schema_version": "huggingface-release-registry/v1",
        "releases": [record],
    }
    with pytest.raises(release.ReleaseError, match="invalid checksum"):
        release.validate_registry_document(document)


def test_checksum_round_trip_and_corruption_detection(tmp_path):
    folder = tmp_path / "release"
    folder.mkdir()
    (folder / "README.md").write_text(dataset_card(), encoding="utf-8")
    (folder / "results.json").write_text('{"status": "valid"}\n', encoding="utf-8")

    written = release.write_checksum_manifest(folder)
    assert written == release.verify_checksum_manifest(folder)

    (folder / "results.json").write_text('{"status": "corrupted"}\n', encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="SHA256 mismatch"):
        release.verify_checksum_manifest(folder)


def test_checksums_reject_symlinks(tmp_path):
    folder = tmp_path / "release"
    folder.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    (folder / "link.txt").symlink_to(target)
    with pytest.raises(release.ReleaseError, match="Symlinks are prohibited"):
        release.compute_checksums(folder)


def test_card_validator_requires_scientific_sections(tmp_path):
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: mit\n---\n\n# Incomplete\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="missing required sections"):
        release.validate_card(card, "dataset")


def test_prepare_dry_run_is_non_mutating(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    card = root / "card.md"
    card.write_text(dataset_card(), encoding="utf-8")
    evidence = root / "evidence" / "result.json"
    evidence.parent.mkdir()
    evidence.write_text('{"status": "valid"}\n', encoding="utf-8")
    spec = root / "release-spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "schema_version": "huggingface-release-spec/v1",
                "id": "artifact-v1",
                "repo_id": "MatthewVaishnav/artifact-v1",
                "repo_type": "dataset",
                "visibility": "private",
                "github_repo": "matthewvaishnav/computational-pathology-research",
                "github_commit": "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce",
                "card": "card.md",
                "source_paths": ["evidence/result.json"],
                "blocked": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "build"
    plan = release.prepare_release(spec, root, output, dry_run=True)
    assert plan["dry_run"] is True
    assert not output.exists()


def test_prepare_creates_provenance_and_verified_checksums(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "card.md").write_text(dataset_card(), encoding="utf-8")
    evidence = root / "evidence" / "result.json"
    evidence.parent.mkdir()
    evidence.write_text('{"status": "valid"}\n', encoding="utf-8")
    spec = root / "release-spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "schema_version": "huggingface-release-spec/v1",
                "id": "artifact-v1",
                "repo_id": "MatthewVaishnav/artifact-v1",
                "repo_type": "dataset",
                "visibility": "private",
                "github_repo": "matthewvaishnav/computational-pathology-research",
                "github_commit": "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce",
                "card": "card.md",
                "source_paths": ["evidence/result.json"],
                "blocked": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "build"
    release.prepare_release(spec, root, output)

    provenance = json.loads(
        (output / "release-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["github_commit"] == "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce"
    assert (output / "evidence/result.json").is_file()
    assert release.verify_checksum_manifest(output)


def test_blocked_panda_spec_fails_closed(tmp_path):
    root = Path(__file__).parents[2]
    spec = (
        root
        / "docs/releases/huggingface/panda-phikon-wsi-spatial-features/release-spec.yaml"
    )
    with pytest.raises(release.ReleaseError, match="Release spec is blocked"):
        release.prepare_release(spec, root, tmp_path / "panda")


def test_public_publish_requires_explicit_flag(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    spec = root / "spec.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "schema_version": "huggingface-release-spec/v1",
                "id": "public-v1",
                "repo_id": "MatthewVaishnav/public-v1",
                "repo_type": "dataset",
                "visibility": "public",
                "github_repo": "matthewvaishnav/computational-pathology-research",
                "github_commit": "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce",
                "card": "card.md",
                "source_paths": ["evidence"],
                "blocked": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    folder = root / "release"
    folder.mkdir()
    (folder / "README.md").write_text(dataset_card(), encoding="utf-8")
    release.write_checksum_manifest(folder)
    registry = root / "registry.yaml"
    registry.write_text(
        "schema_version: huggingface-release-registry/v1\nreleases: []\n"
    )

    with pytest.raises(release.ReleaseError, match="explicit --allow-public"):
        release.publish_release(spec, folder, registry, dry_run=True)

    plan = release.publish_release(
        spec,
        folder,
        registry,
        dry_run=True,
        allow_public=True,
    )
    assert plan["dry_run"] is True
    assert plan["visibility"] == "public"
    assert plan["repo_id"] == "MatthewVaishnav/public-v1"
