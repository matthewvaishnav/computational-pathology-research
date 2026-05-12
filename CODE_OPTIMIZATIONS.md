# Code Optimization Opportunities

## Summary
This document identifies potential performance optimizations across the codebase. These optimizations focus on:
- Using `enumerate()` instead of `range(len())`
- Caching repeated dictionary lookups
- Using list comprehensions instead of append loops
- Memory-efficient generator expressions

## Priority 1: High-Impact Optimizations

### 1. Replace `range(len())` with `enumerate()`

**Impact**: Better readability, slight performance improvement
**Files affected**: 25+ files

#### Example: src/visualization/timeline.py (Line 185)
```python
# Current (inefficient)
for i in range(len(scan_dates)):
    ax.scatter(
        scan_dates[i],
        state_indices[i],
        c=colors[i],
        ...
    )

# Optimized
for i, (date, state_idx, color) in enumerate(zip(scan_dates, state_indices, colors)):
    ax.scatter(
        date,
        state_idx,
        c=color,
        ...
    )
```

#### Other affected files:
- `src/streaming/tracing.py:306`
- `src/streaming/clinical_validation_suite.py:471`
- `src/spatial/tissue_graph.py:198, 211`
- `src/research_platform/annotation_platform.py:322`
- `src/omics/encoders.py:41`
- `src/mobile_edge/compression/calibration_dataset.py:166, 255, 366`
- `src/inference/optimized_inference.py:169, 371`
- `src/federated/pathology_fl_utils.py:240`
- `src/explainability/uncertainty_quantification.py:295`
- `src/federated/aggregator/fedavg.py:70, 107`
- `src/discovery/validation.py:67`
- `src/discovery/subtype.py:58`
- `src/continuous_learning/federated_learning.py:256`
- `src/clinical/reporting.py:818`
- `src/clinical/privacy.py:168`
- `src/clinical_validation/site_leakage_auditor.py:310`
- `src/clinical_validation/performance_metrics.py:216, 366`
- `src/clinical_validation/bias_detection.py:368`
- `src/causal/graphs.py:86, 114`
- `src/annotation_interface/backend/quality_control.py:131`

### 2. Cache Repeated Dictionary Lookups

**Impact**: Reduces redundant lookups, improves performance in hot paths
**Files affected**: 15+ files

#### Example: src/utils/interpretability.py (Line 91)
```python
# Current (inefficient - double lookup)
if embeddings.get(mod_i) is not None and embeddings.get(mod_j) is not None:
    emb_i = embeddings[mod_i].detach().cpu().numpy()
    emb_j = embeddings[mod_j].detach().cpu().numpy()

# Optimized (cache lookups)
emb_i = embeddings.get(mod_i)
emb_j = embeddings.get(mod_j)
if emb_i is not None and emb_j is not None:
    emb_i = emb_i.detach().cpu().numpy()
    emb_j = emb_j.detach().cpu().numpy()
```

#### Example: src/streaming/checkpoint_loader.py (Line 123)
```python
# Current (nested .get() calls)
fe_config = config.get("model", {}).get("feature_extractor", {})
model_name = fe_config.get("model", "resnet50")

# Optimized (cache intermediate result)
model_config = config.get("model", {})
fe_config = model_config.get("feature_extractor", {})
model_name = fe_config.get("model", "resnet50")
```

#### Other affected files:
- `src/streaming/wsi_stream_reader.py:299`
- `src/streaming/cache.py:379`
- `src/pacs/hl7_integration.py:380`
- `src/pacs/clinical_workflow.py:421`
- `src/models/pretrained.py:291`
- `src/integration/emr/epic_fhir_plugin.py:459, 495, 567, 601, 692, 696`
- `src/integration/emr/hl7_message_handler.py:701, 711, 721`
- `src/integration/emr/cerner_emr_plugin.py:407, 443, 478, 542, 641, 669, 671, 674`
- `src/integration/emr/allscripts_emr_plugin.py:528, 738, 739`
- `src/federated/production/monitoring.py:936, 941`

### 3. Use List Comprehensions Instead of Append Loops

**Impact**: More Pythonic, often faster for simple transformations
**Files affected**: 20+ files

#### Example: src/streaming/clinical_report_generator.py (Line 723)
```python
# Current (append in loop)
for i, rec in enumerate(report_data.recommendations, 1):
    elements.append(Paragraph(f"{i}. {rec}", self.styles["Normal"]))

# Optimized (list comprehension)
elements.extend([
    Paragraph(f"{i}. {rec}", self.styles["Normal"])
    for i, rec in enumerate(report_data.recommendations, 1)
])
```

#### Example: src/research_platform/wandb_integration.py (Line 646)
```python
# Current (append in loop)
activity = []
for run in runs:
    activity.append({
        "type": "run_created",
        "run_id": run.id,
        ...
    })

# Optimized (list comprehension)
activity = [
    {
        "type": "run_created",
        "run_id": run.id,
        ...
    }
    for run in runs
]
```

#### Other affected files:
- `src/streaming/clinical_validation.py:488`
- `src/research_platform/dataset_manager.py:309`
- `src/research_platform/annotation_platform.py:205`
- `src/monitoring/ids.py:390`
- `src/monitoring/siem.py:394`
- `src/models/foundation/cache.py:101`
- `src/mobile_edge/compression/gradual_pruning.py:383`
- `src/mobile_edge/compression/lottery_ticket_pruning.py:383`
- `src/interpretability/dashboard.py:688`
- `src/integration/lis/bidirectional_sync.py:606`
- `src/integration/emr/epic_fhir_plugin.py:681`
- `src/hypothesis/generator.py:110, 149`
- `src/foundation/data_collection.py:325`
- `src/federated/production/monitoring.py:356`
- `src/federated/aggregator/byzantine_robust.py:117, 503, 552`
- `src/federated/coordinator/monitoring.py:154`
- `src/explainability/uncertainty_quantification.py:335`

## Priority 2: Medium-Impact Optimizations

### 4. Use Generator Expressions for Large Datasets

**Impact**: Reduces memory usage for large iterations

#### Example Pattern:
```python
# Current (creates full list in memory)
total = sum([expensive_operation(x) for x in large_dataset])

# Optimized (generator - processes one at a time)
total = sum(expensive_operation(x) for x in large_dataset)
```

### 5. Avoid Repeated `any()` Calls

**Impact**: Minor performance improvement in validation code

#### Example: src/utils/password_strength.py (Line 83)
```python
# Current
keyboard_patterns = ['qwerty', 'asdfgh', 'zxcvbn', '1qaz2wsx']
if any(pattern in password.lower() for pattern in keyboard_patterns):
    feedback.append("Avoid keyboard patterns")
    score -= 15

# Optimized (cache lowercase)
password_lower = password.lower()
keyboard_patterns = ['qwerty', 'asdfgh', 'zxcvbn', '1qaz2wsx']
if any(pattern in password_lower for pattern in keyboard_patterns):
    feedback.append("Avoid keyboard patterns")
    score -= 15
```

## Priority 3: Code Quality Improvements

### 6. Simplify Nested Loops with itertools

#### Example: src/spatial/tissue_graph.py
```python
# Current
for i in range(len(centroids)):
    for j in range(i + 1, len(centroids)):
        # process pairs

# Optimized
from itertools import combinations
for i, j in combinations(range(len(centroids)), 2):
    # process pairs
```

### 7. Use zip() for Parallel Iteration

Already well-implemented in many places, but some opportunities remain.

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. Replace all `range(len())` with `enumerate()` or `zip()`
2. Cache repeated dictionary lookups in hot paths
3. Run existing test suite to verify no regressions

### Phase 2: List Comprehensions (2-3 hours)
1. Convert simple append loops to list comprehensions
2. Keep complex loops as-is for readability
3. Run performance benchmarks on critical paths

### Phase 3: Advanced Optimizations (3-4 hours)
1. Profile code to identify actual bottlenecks
2. Apply generator expressions where memory is a concern
3. Consider caching for expensive computations

## Testing Strategy

1. **Unit Tests**: Ensure all existing tests pass
2. **Performance Tests**: Benchmark critical paths before/after
3. **Integration Tests**: Verify end-to-end functionality
4. **Memory Profiling**: Check memory usage improvements

## Estimated Impact

- **Performance**: 5-15% improvement in hot paths
- **Memory**: 10-30% reduction in peak memory for large datasets
- **Code Quality**: Improved readability and Pythonic style
- **Maintainability**: Easier to understand and modify

## Notes

- These optimizations are **safe** and maintain existing functionality
- Focus on **readability** first - don't optimize prematurely
- **Profile before optimizing** - measure actual bottlenecks
- Some patterns (like append loops) may be clearer than comprehensions for complex logic

## Next Steps

1. Review this document with the team
2. Prioritize optimizations based on profiling data
3. Create GitHub issues for each optimization category
4. Implement in small, testable batches
5. Measure and document performance improvements
