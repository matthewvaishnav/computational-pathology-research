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
    body = "\n\n".join(f"## {heading.title()}\n\nExact bounded content." for heading in headings)
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
    records = release.validate_registry(root / "docs/releases/huggingface-release-registry.yaml")
    assert {record["id"] for record in records} >= {
        "panf-evidence-20260726-v1",
        "panf-model-scorpion-capacity-matched-v1",
        "panda-phikon-wsi-spatial-features-300-v1",
        "wsi-nca-phase-a",
        "transnnmil-repaired-canonical",
        "pathologyfl-fair-weights-h",
    }
    release.validate_card(
        root / "docs/releases/huggingface/paired-acquisition-factorization-evidence/README.md",
        "dataset",
    )
    release.validate_card(
        root / "docs/releases/huggingface/panda-phikon-wsi-spatial-features/README.md",
        "dataset",
    )
    release.validate_card(
        root / "docs/releases/huggingface/paired-acquisition-neural-factorization/README.md",
        "model",
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
    link = folder / "link.txt"
    try:
        link.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    except NotImplementedError:
        pytest.skip("Symlink creation is unavailable on this platform")
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

    provenance = json.loads((output / "release-provenance.json").read_text(encoding="utf-8"))
    assert provenance["github_commit"] == "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce"
    assert (output / "evidence/result.json").is_file()
    assert release.verify_checksum_manifest(output)


def test_blocked_panda_spec_fails_closed(tmp_path):
    root = Path(__file__).parents[2]
    spec = root / "docs/releases/huggingface/panda-phikon-wsi-spatial-features/release-spec.yaml"
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
    registry.write_text("schema_version: huggingface-release-registry/v1\nreleases: []\n")

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


def test_registry_update_after_publish_clears_resolved_blocker(tmp_path):
    record = registry_record()
    record["visibility"] = "public"
    record["blocker"] = "Hugging Face authentication is unavailable"

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": "huggingface-release-registry/v1",
                "last_verified_at": "2026-08-12T16:30:00Z",
                "releases": [record],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    checksums = {"evidence.json": "a" * 64}

    release.update_registry_after_publish(
        registry,
        "artifact-v1",
        revision="abcdef0123456789",
        checksums=checksums,
        visibility="public",
    )

    updated = release.load_yaml(registry)
    released = updated["releases"][0]

    assert released["release_state"] == "released"
    assert released["hf_revision"] == "abcdef0123456789"
    assert released["checksums"] == checksums
    assert released["blocker"] is None
    assert updated["last_verified_at"] == released["last_verified_at"]


def test_public_model_bundle_fails_closed_on_license_gate(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    checkpoint = folder / "checkpoints/fold_0/seed_801/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    cell_manifest = folder / "provenance/cell.json"
    cell_manifest.parent.mkdir(parents=True)
    cell_manifest.write_text("{}\n", encoding="utf-8")
    standardization = folder / "preprocessing/fold_0_standardization.npz"
    standardization.parent.mkdir(parents=True)
    standardization.write_bytes(b"standardization")
    manifest = {
        "schema_version": "panf-model-transfer-bundle/v1",
        "release_id": "panf-model-test-v1",
        "repo_id": "MatthewVaishnav/panf-model-test-v1",
        "checkpoints": [
            {
                "fold": 0,
                "seed": 801,
                "checkpoint_path": checkpoint.relative_to(folder).as_posix(),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": release.sha256_file(checkpoint),
                "source_cell_manifest_path": cell_manifest.relative_to(folder).as_posix(),
                "source_cell_manifest_size_bytes": cell_manifest.stat().st_size,
                "source_cell_manifest_sha256": release.sha256_file(cell_manifest),
                "content_validation": {
                    "torch_load": True,
                    "metadata": True,
                    "config": True,
                    "state_dict_keys_and_shapes": True,
                    "finite_tensors": True,
                },
            }
        ],
        "preprocessing": [
            {
                "fold": 0,
                "path": standardization.relative_to(folder).as_posix(),
                "size_bytes": standardization.stat().st_size,
                "sha256": release.sha256_file(standardization),
            }
        ],
        "license_gate": {
            "public_release_allowed": False,
            "reason": "source-data redistribution permission is unresolved",
        },
    }
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec = {
        "schema_version": "huggingface-release-spec/v1",
        "id": "panf-model-test-v1",
        "repo_id": "MatthewVaishnav/panf-model-test-v1",
        "repo_type": "model",
        "visibility": "public",
        "github_repo": "matthewvaishnav/computational-pathology-research",
        "github_commit": "edf8b2b96fbdb8b21fbecc03b8a19ac0351e1dce",
        "card": "card.md",
        "source_paths": ["checkpoint.pt"],
        "external_bundle": {
            "schema_version": "panf-model-transfer-bundle/v1",
            "expected_checkpoints": 1,
            "expected_preprocessing_files": 1,
            "manifest": "model-manifest.json",
        },
    }
    with pytest.raises(release.ReleaseError, match="license gate"):
        release.validate_model_release_folder(folder, spec)

    manifest["license_gate"]["public_release_allowed"] = True
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = release.validate_model_release_folder(folder, spec)
    assert result == {
        "checkpoints": 1,
        "preprocessing_files": 1,
        "identities": 1,
        "license_gate": True,
    }
