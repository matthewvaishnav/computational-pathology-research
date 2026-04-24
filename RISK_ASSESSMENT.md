# Risk Assessment and Fixes

## High Priority Risks

### 1. 🟡 src/cells/detector.py L159-160
**Risk**: skimage import in function body + scipy fallback without watershed
**Impact**: Performance hit on every call, behavior difference between skimage/scipy
**Severity**: Medium-High

### 2. 🟡 src/cells/graph.py L115
**Risk**: regionprops on single-label mask uses unnecessary sk_label
**Impact**: Incorrect results - mask is already binary, sk_label creates new labels
**Severity**: High

### 3. 🟡 src/cells/graph.py L162
**Risk**: Delaunay exception too broad, catches all exceptions
**Impact**: Silently swallows errors, hard to debug
**Severity**: Medium

### 4. 🟡 src/cells/gnn.py L73
**Risk**: _global_mean_pool_fallback assumes batch is contiguous 0..B-1
**Impact**: Incorrect pooling if batch indices have gaps
**Severity**: High

### 5. 🟡 src/hypothesis/generator.py L95
**Risk**: No timeout on API call
**Impact**: Hangs indefinitely on network issues
**Severity**: Medium

### 6. 🟡 src/hypothesis/generator.py L104
**Risk**: text.split("```")[1] assumes exactly 2 fences
**Impact**: IndexError if format is unexpected
**Severity**: Medium

### 7. 🟡 src/causal/estimators.py L72
**Risk**: IPW variance explodes near e=0 or e=1
**Impact**: Unstable estimates despite trimming
**Severity**: Medium

### 8. 🟡 src/federated/privacy.py L165
**Risk**: noise_std broadcast to full param.grad causes memory spike
**Impact**: OOM on large models
**Severity**: Medium-High

### 9. 🟡 src/spatial/alignment.py L186
**Risk**: adata.X.toarray() loads full sparse matrix
**Impact**: OOM on large datasets
**Severity**: Medium-High

### 10. 🟡 src/omics/fusion.py L82
**Risk**: masked_fill with -inf before softmax, all-masked rows → NaN
**Impact**: NaN propagation in forward pass
**Severity**: High

## Fixing Strategy

1. **Immediate fixes** (High severity, low complexity)
2. **Performance fixes** (Medium-High severity, memory issues)
3. **Robustness fixes** (Medium severity, error handling)
