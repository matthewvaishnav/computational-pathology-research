# Hugging Face release tooling

This directory implements the curated release layer described in
`docs/releases/huggingface-release-registry.yaml`. GitHub remains authoritative
for source, tests, experiments, evidence, negative results, and repair history.

The tool is fail-closed:

- credentials come only from `HF_TOKEN` or the official Hub credential store;
- release specs with unresolved blockers cannot be prepared or published;
- public publishing requires `--allow-public`;
- private repositories are never made public by this tool;
- cards, source paths, exact Git commits, local checksums, and remote checksums
  are validated before the registry is updated;
- release folders cannot contain symlinks.

Install the small release-only dependency set:

```bash
python -m pip install -r tools/huggingface/requirements.txt
```

Validate the registry and grounded cards:

```bash
python tools/huggingface/release.py validate-registry \
  docs/releases/huggingface-release-registry.yaml
python tools/huggingface/release.py validate-card \
  docs/releases/huggingface/paired-acquisition-factorization-evidence/README.md \
  --type dataset
```

Dry-run the public evidence release preparation:

```bash
python tools/huggingface/release.py prepare \
  docs/releases/huggingface/paired-acquisition-factorization-evidence/release-spec.yaml \
  --repository-root . \
  --output build/huggingface/paired-acquisition-factorization-evidence \
  --dry-run
```

After review, build, publish, and verify it. The explicit public flag is
required because a public repository is an irreversible disclosure boundary:

```bash
python tools/huggingface/release.py prepare \
  docs/releases/huggingface/paired-acquisition-factorization-evidence/release-spec.yaml \
  --repository-root . \
  --output build/huggingface/paired-acquisition-factorization-evidence
python tools/huggingface/release.py publish \
  docs/releases/huggingface/paired-acquisition-factorization-evidence/release-spec.yaml \
  build/huggingface/paired-acquisition-factorization-evidence \
  --registry docs/releases/huggingface-release-registry.yaml \
  --allow-public
```

The PANDA spec remains blocked until the real 300-slide bundle is present and
the non-commercial/share-alike redistribution basis is explicitly accepted.
It cannot be published accidentally.

## PA-NF trained model family

The registered PA-NF release is a complete 25-checkpoint family plus five
fold-specific standardizers. On the Windows machine containing the campaign
`results` directory, run this from the repository root in PowerShell:

```powershell
python tools/huggingface/panf_model_bundle.py build --artifact-index evidence/paired_acquisition/scorpion-capacity-matched-20260726/campaign/cell_artifact_index.csv --source-results-root .\results --target-bundle build\huggingface\paired-acquisition-neural-factorization
```

The build is atomic and refuses missing cells, substituted runs, hash mismatches,
invalid checkpoint metadata/config/state shapes, inconsistent preprocessing
hashes, and copied-byte mismatches. Re-verify the completed folder with:

```powershell
python tools/huggingface/panf_model_bundle.py verify --bundle build\huggingface\paired-acquisition-neural-factorization
python tools/huggingface/release.py verify-local build\huggingface\paired-acquisition-neural-factorization
```

The PA-NF model release is licensed under Apache-2.0. The model package includes
the 25 registered checkpoints, the five required fold standardizers, the exact
model definition, release-specific inference code, provenance manifests, model
card, Apache-2.0 license file, and checksums. Raw SCORPION images and the frozen
DINOv2 feature archive are excluded and retain their own upstream terms.

After rebuilding from the latest branch state and verifying the folder, publish
through the standard fail-closed path:

```powershell
python tools/huggingface/release.py publish `
  docs/releases/huggingface/paired-acquisition-neural-factorization/release-spec.yaml `
  build/huggingface/paired-acquisition-neural-factorization `
  --registry docs/releases/huggingface-release-registry.yaml `
  --allow-public
```

The publisher validates the complete model-family inventory before upload,
redownloads the remote repository, verifies every file checksum, and records the
immutable Hugging Face revision in the release registry only after verification.
