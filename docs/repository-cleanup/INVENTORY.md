# Repository cleanup inventory

This document records the first evidence-based classification of the tracked repository at cleanup audit commit `ed6a339bd9472f2bb0d2eab0e8948f5ee872a712`.

The classification is intentionally conservative. A path marked **review/relocate** or **removal candidate** must not be deleted until its imports, workflow references, documentation links, and reproducibility role have been checked.

## Audit snapshot

- 2,114 tracked files totaling 49.13 MiB.
- 1,154 Python files, 336 Markdown files, 180 CSV files, and 106 JSON files.
- 15 exact duplicate-content groups.
- Three Python syntax failures:
  - `src/streaming/cache.py`: truncated Redis constructor;
  - `scripts/download_pmc_pathology.py`: repeated `timeout` keyword;
  - `scripts/histocore-admin.py`: repeated `timeout` keyword.
- Two explicitly suspicious tracked files:
  - `docs/VERIFIED_METRICS.json` (8.64 MiB);
  - `paper/arxiv/build/main.tex` (generated paper build output).
- Generated coverage/result snapshots account for several of the largest duplicate files.
- The frozen PathoAlign two-resource evidence chain is tracked and must remain reproducible.

## Cleanup rules

1. Work only on `repo-cleanup-2026-06-10`; keep PR #7 draft until the final diff, CI, paper build, and public PDF URL are verified.
2. Preserve source inputs, compact derived evidence, analysis code, tests, and paper claims that form a reproducibility chain.
3. Do not preserve generated output merely because it is already tracked. Generated artifacts need a documented reason to remain.
4. Do not treat a skipped test as validation. Duplicate or obsolete tests must be consolidated or archived with an explanation.
5. Do not weaken CI to hide failures. Split CI into fast core, optional subsystem, slow integration, and paper/reproducibility tiers.
6. Do not make unconditional acquisition-component subtraction the default PathoAlign method; preserve the tumor-information leakage audits and the conditional low-rank direction.
7. Never change the deployed PDF path `when-more-data-is-less-trustworthy-references-v2.pdf`.

## Top-level directory classification

| Path | Audit evidence | Working classification | Required action |
|---|---:|---|---|
| `.github/` | 19 files; 16 workflows | active project infrastructure | Keep the audit workflow temporarily; tier overlapping CI and preserve the paper deployment contract. |
| `assets/` | 1 small tracked asset | paper/documentation | Identify its consumer and relocate under `docs/assets/` or `paper/` if it has no root-level role. |
| `benchmarks/` | 5 files | reproducibility evidence / optional subsystem | Keep only benchmarks with runnable commands, manifests, and stated hardware/software assumptions. |
| `business/` | 1 file | historical archive or obsolete/removable | Inspect for unsupported commercial/clinical claims; archive outside active research navigation or remove. |
| `checkpoints/` | only `.gitkeep` | generated output | Remove the tracked placeholder if checkpoint paths are already ignored and documented. |
| `cloud/` | 17 AWS/Azure/GCP deployment files | optional subsystem / likely historical archive | Verify whether any workflow exercises these files; do not present them as validated deployment infrastructure. |
| `config/` | 3 production/PACS/model-revision files | optional subsystem | Distinguish production-era HistoCore configuration from active research configuration in `configs/`. |
| `configs/` | 44 experiment/model/task configs | active research experiment | Preserve active Camelyon17, PANDA, federated, and PathoAlign configs; add ownership and entry-point mapping. |
| `coverage_reports/` | 0.58 MiB generated JSON | generated output | Remove from tracking after confirming no documentation consumes it; keep coverage generation in CI artifacts. |
| `data/` | 4 small manifest/placeholder files | active research data metadata | Preserve manifests and README; keep datasets and derived arrays untracked. |
| `dataset_test_results/` | 13 dated summaries/results including duplicate coverage | generated output / historical evidence | Retain at most one compact, justified snapshot or move to archive; remove redundant timestamped copies. |
| `deploy/` | 3 API deployment files | optional subsystem | Verify imports and tests; classify as experimental rather than production-ready unless actively validated. |
| `docker/` | 16 API/federated/monitoring files | optional subsystem | Consolidate with root Docker files; keep only buildable images used by a maintained workflow. |
| `docs/` | 247 files, 13.24 MiB | paper/documentation plus historical archive | Split active research docs from legacy platform/clinical/deployment claims; deduplicate mirrored pages and remove generated metrics blobs. |
| `ecosystem/` | 1 file | historical archive or obsolete/removable | Inspect for unsupported ecosystem claims and remove from active navigation unless it reflects current integrations. |
| `enterprise/` | 2 demo/ROI files | historical archive or obsolete/removable | Remove commercial demo framing from the active research tree or explicitly archive it. |
| `examples/` | 51 files | optional subsystem / documentation | Keep only examples that import current APIs and can be smoke-tested; archive stale clinical/platform demos. |
| `experiments/` | 120 files | active research experiment mixed with historical scaffolding | Separate reproducible current experiments from benchmark-system and v2.0 history; map each active experiment to configs, scripts, and results. |
| `figures/` | 4 paper/research figures | reproducibility evidence / paper | Preserve when referenced by the paper or reports; document generation commands where available. |
| `fl_audit_logs/` | 2 JSONL logs | reproducibility evidence or generated output | Check for sensitive or synthetic content and paper references; retain only compact anonymized evidence if justified. |
| `fl_checkpoints/` | 1 small file | generated output | Inspect whether it is a real model state; model checkpoints should normally be release assets or external artifacts, not source-controlled. |
| `k8s/` | 50 Kubernetes/Helm files | optional subsystem / likely historical archive | Verify deployability and remove unsupported production claims; consolidate with `kubernetes/`. |
| `kubernetes/` | 1 file | duplicate/misplaced optional subsystem | Merge into the canonical deployment location or archive. |
| `migrations/` | 3 Alembic files | optional subsystem | Keep only if the active package still exposes a database-backed service and tests its schema. |
| `models/` | only `.gitkeep` | generated output | Remove placeholder after confirming model outputs are ignored. |
| `monitoring/` | 12 Prometheus/Grafana files | optional subsystem / likely historical archive | Deduplicate Grafana/Prometheus configuration with `docker/`; retain only maintained dashboards. |
| `notebooks/` | 5 notebooks/placeholders | documentation / historical archive | Execute or clearly mark notebooks as historical; remove empty placeholder if unnecessary. |
| `panda/` | dataset CSV and split JSON | active research data metadata | Verify licensing and whether the full CSV belongs in Git; preserve deterministic split metadata where legally appropriate. |
| `paper/` | 7 source/build files | paper/documentation | Preserve source, bibliography, calculations, build script, filename, and public URL; remove generated `build/main.tex` from tracking if reproducible. |
| `patents/` | 1 file | historical archive or obsolete/removable | Remove from active scientific navigation unless there is a clear repository purpose. |
| `results/` | 267 files, 9.67 MiB | reproducibility evidence mixed with generated output | Preserve compact frozen evidence chains; inventory every result family and remove raw predictions/repeated run outputs when summaries and manifests suffice. |
| `scripts/` | 175 files | active research experiment mixed with historical tooling | Repair syntax first; group active data, training, Camelyon17, federated, and PathoAlign entry points; archive HistoCore administration/deployment scripts. |
| `src/` | 554 files, 6.46 MiB | active package source mixed with optional and legacy subsystems | Repair imports and cache truncation; identify the minimal active research package and isolate clinical/platform/streaming/deployment compatibility surfaces. |
| `test_results/` | 3 generated files | generated output | Remove from tracking after confirming no active consumer; publish future outputs as CI artifacts. |
| `tests/` | 391 files, 8.16 MiB | active tests mixed with obsolete/duplicate suites | Consolidate duplicate cache tests, separate optional dependencies, and make the smallest core suite authoritative. |
| `viz/` | 1 generated PNG | generated output / documentation | Keep only if referenced; otherwise regenerate from source into CI or paper artifacts. |
| `website/` | 42 Docusaurus files, 3.80 MiB | historical/duplicate documentation surface | Determine whether VitePress under `docs/` is canonical; avoid maintaining two sites and duplicated static assets. |

## Root-level file groups

| Group | Working classification | Required action |
|---|---|---|
| `README.md`, `CITATION.cff`, `CLAIM_BOUNDARY.md`, `LICENSE`, `SECURITY.md` | active project metadata | Align identity and claim boundaries with the actual computational-pathology research program. |
| `FEDERATED_TEST_*`, `MILESTONE_*`, `TASK_LOG.md`, `RECRUITER_README.md`, `CHANGELOG.md` | mixed historical status documentation | Move dated status material into `docs/archive/` or consolidate into current research navigation. |
| `pyproject.toml`, `VERSION`, requirements files | active package/build metadata | Replace stale HistoCore identity and unsupported deployment/test-count claims; rationalize overlapping requirement sets. |
| Docker Compose, Dockerfile, installation scripts, `alembic.ini`, `.env*` | optional subsystem | Retain only if paired with maintained, tested services; otherwise archive. |
| `package.json`, `package-lock.json`, `mkdocs.yml` | documentation tooling | Remove abandoned tooling after choosing one documentation stack. |

## Protected reproducibility chain

The following paths are protected from cleanup deletion or silent rewriting until the corresponding analysis is reproduced:

- `results/pathoalign_two_resource_analysis/`
- `results/pathoalign_two_resource_phase_map/`
- `scripts/pathoalign_identifiability_v6/analyze_two_resource_law.py`
- `scripts/pathoalign_identifiability_v6/fit_censored_threshold_models.py`
- `tests/test_pathoalign_v6_analysis.py`
- `paper/arxiv/identifiability_calculations.tex`
- `paper/arxiv/main.tex`
- `paper/arxiv/references.bib`
- `.github/workflows/vitepress.yml`

## First repair tranche

1. Restore `src/streaming/cache.py` from the last complete historical implementation, then reconcile later security/error-handling changes.
2. Remove the repeated `timeout` keywords in the two broken scripts.
3. Run syntax/import checks before broader tests.
4. Consolidate the two cache test generations only after the restored implementation is validated.
5. Correct package identity before deleting broad legacy infrastructure, so compatibility surfaces are explicit.
