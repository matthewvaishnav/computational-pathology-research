# Repository Cleanup Plan

## Current Problem
Repository has 100+ root-level files and directories, making it cluttered and unprofessional.

## Target Clean Structure

```
computational-pathology-research/
├── .github/              # CI/CD workflows
├── .kiro/                # Kiro AI config
├── docs/                 # Documentation
├── examples/             # Example scripts
├── experiments/          # Experiment configs
├── scripts/              # Utility scripts
├── src/                  # Source code (already clean)
├── tests/                # Test suite
├── k8s/                  # Kubernetes manifests
├── docker/               # Docker configs
├── website/              # Documentation website
├── .gitignore
├── .dockerignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
├── SECURITY.md
└── (essential config files)
```

## Cleanup Actions

### 1. DELETE - Temporary/Generated Files (Safe to remove)
```
bandit*.json (15 files)          # Old security scan results
safety_results.json
security_scan_results.json
security-posture-final.*
coverage.xml
htmlcov/
.coverage
.pytest_cache/
output.txt
overnight_training.log
demo_ablation_results/
demo_output/
outputs/
logs/
```

### 2. DELETE - Migration Scripts (No longer needed)
```
update_imports_phase*.py (6 files)
complete_migration_phases7-10.py
fix_relative_imports_phase3.py
PHASE1_COMPLETE.md
PHASE2_COMPLETE.md
PHASE3_COMPLETE.md
MIGRATION_PROGRESS.md
```

### 3. DELETE - Test Scripts (Move to tests/ or delete)
```
test_*.py (8 root-level test files)
check_gpu.py
quick_train_test.py
train_transnnmil_pcam.py
verify_feature_fusion.py
setup_panda_after_download.py
```

### 4. DELETE - Virtual Environments (Should be in .gitignore)
```
.venv/
.venv-ci311/
.venv311/
venv/
venv_gpu/
venv311/
envs/
```

### 5. DELETE - IDE/Tool Configs (Keep in .gitignore)
```
.vscode/
.qodo/
.hypothesis/
.kilo/
opus-delegation/
opus-delegation-system/
```

### 6. MOVE - Data/Results to data/
```
checkpoints/          → data/checkpoints/
panda/                → data/panda/
models/               → data/pretrained_models/
coverage_reports/     → data/coverage_reports/
dataset_test_results/ → data/dataset_test_results/
test_results/         → data/test_results/
results/              → data/results/
fl_checkpoints/       → data/fl_checkpoints/
pacs_cache/           → data/pacs_cache/
```

### 7. MOVE - Documentation
```
ARCHITECTURE_MIGRATION.md  → docs/architecture/MIGRATION.md
MIGRATION_COMPLETE.md      → docs/architecture/MIGRATION_COMPLETE.md
GIT_UPDATE_SUMMARY.md      → docs/GIT_UPDATE_SUMMARY.md
GITHUB_DESCRIPTION.txt     → docs/GITHUB_DESCRIPTION.txt
UPDATE_GITHUB_REPO.md      → docs/UPDATE_GITHUB_REPO.md
REPO_ORGANIZATION.md       → docs/REPO_ORGANIZATION.md
REPOSITORY_OVERVIEW.md     → docs/REPOSITORY_OVERVIEW.md
PANDA_QUICK_START.md       → docs/PANDA_QUICK_START.md
PANDA_SETUP_GUIDE.md       → docs/PANDA_SETUP_GUIDE.md
TRAINING_SETUP.md          → docs/TRAINING_SETUP.md
TRANSFER_CHECKLIST.md      → docs/TRANSFER_CHECKLIST.md
MODEL_CARD_V2.md           → docs/MODEL_CARD_V2.md
NEXT_STEPS_OTHER_PC.md     → docs/NEXT_STEPS_OTHER_PC.md
```

### 8. MOVE - Deployment/Infrastructure
```
cloud/                → deploy/cloud/
deploy/               → deploy/basic/
kubernetes/           → k8s/ (merge with existing)
migrations/           → deploy/migrations/
alembic.ini           → deploy/alembic.ini
```

### 9. MOVE - Business/Enterprise
```
business/             → docs/business/
enterprise/           → docs/enterprise/
patents/              → docs/patents/
ecosystem/            → docs/ecosystem/
```

### 10. MOVE - Monitoring/Audit
```
monitoring/           → data/monitoring/
fl_audit_logs/        → data/fl_audit_logs/
```

### 11. MOVE - Assets/Viz
```
assets/               → docs/assets/
viz/                  → docs/viz/
test_docs/            → docs/test_docs/
```

### 12. CONSOLIDATE - Config Files (Keep at root, but fewer)
```
Keep:
- .gitignore
- .dockerignore
- .flake8
- .isort.cfg
- .coveragerc
- .mailmap
- .pre-commit-config.yaml
- pyproject.toml
- requirements*.txt
- docker-compose*.yml
- Dockerfile

Move to config/:
- .env.example        → config/env.example
- .env.docker         → config/env.docker
- .env.production.example → config/env.production.example
- config/             → config/ (keep)
- configs/            → config/ (merge)
```

### 13. KEEP at Root (Essential)
```
- README.md
- LICENSE
- SECURITY.md
- CITATION.cff
- VERSION
- pyproject.toml
- requirements.txt
- docker-compose.yml
- Dockerfile
- install.sh
- install.bat
```

## Implementation Order

1. **Backup first**: Create git branch
2. **Delete temp files**: Safe, no dependencies
3. **Delete migration scripts**: No longer needed
4. **Move data directories**: Update .gitignore
5. **Move documentation**: Update links
6. **Move deployment files**: Update scripts
7. **Consolidate configs**: Update references
8. **Update .gitignore**: Add all data/ patterns
9. **Test**: Ensure nothing breaks
10. **Commit**: Clean repository

## Expected Result

Root directory will have ~20 items instead of 100+:
- 6 directories (docs, examples, experiments, scripts, src, tests, k8s, docker, website, data, deploy, config)
- ~10 essential files (README, LICENSE, requirements, etc.)
- Hidden config files (.gitignore, .flake8, etc.)

## Benefits

- Professional appearance
- Easy to navigate
- Clear separation of concerns
- Easier onboarding for new contributors
- Better for GitHub presentation
