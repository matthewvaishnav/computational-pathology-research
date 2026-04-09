# Repository Improvement Plan

Based on external review feedback, this plan addresses the gap between infrastructure quality and scientific validation.

## Status: Infrastructure is Production-Grade ✅

The repository has:
- ✅ Professional structure (src/, experiments/, tests/, configs/, deploy/, k8s/, docs/)
- ✅ Real benchmarks (PatchCamelyon, CAMELYON16)
- ✅ Mature tooling (OpenSlide, ONNX export, Docker, K8s, pre-commit hooks)
- ✅ Comprehensive CI/CD (GitHub Actions with 9 jobs)
- ✅ 62% test coverage
- ✅ Real PCam data downloaded (327K images in `data/pcam_real/`)
- ✅ Training pipeline verified (3.8 it/s, ~18 min/epoch on RTX 4070 Laptop)

## Gap: Scientific Validation Needs to Catch Up

Current limitation: README leads with synthetic benchmark results, which undermines the strong infrastructure.

## Priority Actions

### 1. Complete Real PCam Benchmark Run (HIGHEST PRIORITY)
**Status**: Training infrastructure ready, real data downloaded
**Action**: Complete the full 20-epoch training run on real PCam data
**Impact**: Transforms from "framework with fake benchmarks" to "framework with real results"
**Timeline**: ~6 hours (20 epochs × 18 min/epoch)

**Steps**:
- [ ] Resume or restart training with `experiments/configs/pcam_rtx4070_laptop.yaml`
- [ ] Run full evaluation on test set
- [ ] Generate metrics with bootstrap confidence intervals
- [ ] Update `docs/PCAM_BENCHMARK_RESULTS.md` with real results
- [ ] Update README with real benchmark numbers

### 2. Reframe README Lead (HIGH PRIORITY)
**Status**: README currently leads with synthetic results
**Action**: Restructure README to lead with framework capabilities
**Impact**: Positions repo as infrastructure project, not just benchmark results

**Changes**:
```markdown
# Before (Current)
> **Research Framework**: Tested implementations for computational pathology 
> with working benchmarks on PatchCamelyon...
> 
> ## Quick Start
> ### PatchCamelyon (PCam) Training
> **Results** (synthetic subset):
> - Test Accuracy: 94.0%
> - Test AUC: 1.0

# After (Proposed)
> **Production-Grade ML Research Framework** for computational pathology
> 
> Provides tested infrastructure for:
> - Whole-slide image (WSI) processing with OpenSlide
> - Multiple Instance Learning (MIL) for slide-level classification
> - Benchmark pipelines for PatchCamelyon and CAMELYON16
> - Model profiling, ONNX export, and deployment tools
> 
> ## Real Benchmark Results
> 
> ### PatchCamelyon (262K train, 32K test)
> - Test Accuracy: XX.X% ± X.X% (95% CI)
> - Test AUC: X.XXX ± X.XXX
> - Hardware: RTX 4070 Laptop (8GB VRAM)
> - Training Time: ~6 hours (20 epochs)
> 
> ## Development/Testing
> Synthetic data generators available for pipeline validation...
```

### 3. Fix CI Badges (MEDIUM PRIORITY)
**Status**: CI configured but badges not in README
**Action**: Add dynamic badges to README

**Badges to add**:
```markdown
[![CI](https://github.com/matthewvaishnav/computational-pathology-research/workflows/CI/badge.svg)](https://github.com/matthewvaishnav/computational-pathology-research/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/matthewvaishnav/computational-pathology-research/branch/main/graph/badge.svg)](https://codecov.io/gh/matthewvaishnav/computational-pathology-research)
```

**Note**: Codecov badge already exists in README but may need token configuration

### 4. Clean IDE Artifacts (MEDIUM PRIORITY)
**Status**: `.kiro/settings/mcp.json` is tracked (IDE config)
**Action**: Remove IDE-specific files, keep spec documentation

**Keep** (legitimate project docs):
- `.kiro/specs/*/requirements.md`
- `.kiro/specs/*/design.md`
- `.kiro/specs/*/tasks.md`
- `.kiro/specs/*/.config.kiro`

**Remove** (IDE-specific):
- `.kiro/settings/mcp.json`

**Commands**:
```bash
# Remove IDE settings from tracking
git rm --cached .kiro/settings/mcp.json

# Update .gitignore
echo ".kiro/settings/" >> .gitignore

# Commit
git commit -m "Remove IDE-specific settings from tracking"
```

### 5. Update CITATION.cff (LOW PRIORITY)
**Status**: Year 2026 with no paper yet
**Action**: Add note about framework citation vs paper citation

**Proposed change**:
```yaml
# CITATION.cff
message: "If you use this framework in your research, please cite it as below. For the associated paper, see [link when available]."
title: "Computational Pathology Research Framework"
version: "0.1.0"
date-released: "2024-XX-XX"  # Use actual release date
```

### 6. Pin Repository on Profile (LOW PRIORITY)
**Action**: Pin this repo on GitHub profile
**Impact**: Increases visibility

## Implementation Order

1. **Week 1**: Complete real PCam benchmark run (Priority 1)
2. **Week 1**: Reframe README (Priority 2)
3. **Week 1**: Fix CI badges (Priority 3)
4. **Week 2**: Clean IDE artifacts (Priority 4)
5. **Week 2**: Update CITATION.cff (Priority 5)
6. **Anytime**: Pin repository (Priority 6)

## Success Metrics

- [ ] Real PCam results in README (not synthetic)
- [ ] README leads with framework capabilities
- [ ] Green CI badge showing in README
- [ ] Dynamic coverage badge (not hardcoded)
- [ ] No IDE-specific files in git history
- [ ] Repository pinned on profile

## Notes

- The infrastructure quality is genuinely impressive
- The weak point is that science hasn't caught up to engineering yet
- This is a roadmap, not a criticism
- Real data is already downloaded - just need to complete the run
