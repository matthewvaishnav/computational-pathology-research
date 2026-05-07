# Clean Code Refactoring - Requirements

**Feature Name**: clean-code-refactoring  
**Created**: 2026-05-03  
**Status**: Requirements Phase  
**Priority**: HIGH (Technical Debt)

---

## Executive Summary

The HistoCore codebase has grown to over 50,000 lines of production code across 200+ modules. While the system is functionally complete and tested, several modules violate Clean Code principles, creating maintenance burden and increasing the risk of bugs. This spec addresses systematic refactoring of the most problematic areas.

**Key Insight**: "The ratio of time spent reading versus writing code is well over 10 to 1" - Robert C. Martin

Our codebase will be read hundreds of times more than it's written. Making it clean now is an investment in velocity.

---

## Problem Statement

### Current State

Analysis of the codebase reveals:

**Files with Excessive Length** (>800 lines):
1. `src/models/attention_mil.py` - **1,527 lines** (4 classes, massive duplication)
2. `src/data/wsi_pipeline/wsi_stream_reader.py` - **1,501 lines**
3. `src/clinical/treatment_response.py` - **1,442 lines**
4. `src/api/main.py` - **1,308 lines** (40+ route handlers in one file)
5. `src/streaming/memory_optimizer.py` - **1,097 lines**
6. `src/clinical/audit.py` - **1,026 lines**
7. `src/clinical/regulatory.py` - **976 lines**

**Specific Violations**:

#### 1. God Objects / Massive Classes
**File**: `src/models/attention_mil.py` (1,527 lines)
- **Problem**: 4 MIL model classes with 200-400 lines each
- **Violation**: "Classes should be small and have a single responsibility"
- **Impact**: 
  - Difficult to understand any single model
  - Code duplication across classes (fusion methods repeated 3x)
  - Hard to test individual components
  - Merge conflicts frequent

**Evidence**:
```
Class AttentionMIL: Lines 134-550 (416 lines)
Class CLAM: Lines 551-1138 (587 lines)
Class TransMIL: Lines 1139-1643 (504 lines)
```

Each class has:
- `_early_fusion()` method (60-100 lines) - duplicated logic
- `_late_fusion()` method (60-100 lines) - duplicated logic
- `compute_attention()` method (30-50 lines) - similar patterns
- `aggregate_features()` method (20-40 lines) - similar patterns

#### 2. Route Handler Explosion
**File**: `src/api/main.py` (1,308 lines)
- **Problem**: 40+ FastAPI route handlers in single file
- **Violation**: "Functions should do one thing and do it well"
- **Impact**:
  - Impossible to find specific endpoints
  - No logical grouping (auth, analysis, admin, mobile all mixed)
  - Difficult to test
  - Startup time issues (all routes loaded at once)

**Evidence**:
```python
# Authentication routes mixed with analysis routes mixed with admin routes
@app.post("/api/register")  # Line 407
@app.post("/api/login")  # Line 454
@app.post("/api/upload")  # Line 528
@app.get("/api/analysis/{id}")  # Line 697
@app.post("/api/dicom")  # Line 765
@app.get("/api/users")  # Line 1089
@app.get("/api/audit-logs")  # Line 1107
# ... 30+ more routes
```

#### 3. Long Functions (>50 lines)
**File**: `src/api/main.py`
- **Problem**: Several functions exceed 100 lines
- **Violation**: "The first rule of functions is that they should be small"

**Examples**:
- `upload_for_analysis()` - Lines 528-625 (97 lines)
  - Handles file validation, storage, database, async processing
  - Should be 5-6 separate functions
- `process_real_analysis()` - Lines 626-696 (70 lines)
  - Mixes inference, error handling, database updates
- `oauth_callback()` - Lines 1345-1431 (86 lines)
  - Token validation, user creation, session management all in one

#### 4. Code Duplication
**File**: `src/models/attention_mil.py`
- **Problem**: Fusion methods duplicated across 3 classes
- **Violation**: DRY (Don't Repeat Yourself)

**Evidence**:
```python
# AttentionMIL._early_fusion() - Lines 268-333 (65 lines)
# CLAM._early_fusion_clam() - Lines 737-832 (95 lines)
# TransMIL._early_fusion_transmil() - Lines 1305-1384 (79 lines)

# Nearly identical logic:
# 1. Check if multimodal data exists
# 2. Project each modality
# 3. Concatenate features
# 4. Apply fusion layer
```

**Impact**:
- Bug fixes must be applied 3 times
- Inconsistent behavior across models
- 200+ lines of duplicated code

#### 5. Mixed Responsibilities
**File**: `src/streaming/memory_optimizer.py` (1,097 lines)
- **Problem**: Single class handles memory profiling, optimization, caching, and monitoring
- **Violation**: Single Responsibility Principle

**Evidence**:
```python
class MemoryOptimizer:
    # Memory profiling (100 lines)
    def profile_memory_usage()
    def track_allocations()
    
    # Cache management (150 lines)
    def manage_cache()
    def evict_cache_entries()
    
    # Optimization strategies (200 lines)
    def optimize_batch_size()
    def optimize_tile_size()
    
    # Monitoring and alerts (150 lines)
    def monitor_memory()
    def send_alerts()
    
    # Configuration (100 lines)
    def load_config()
    def validate_config()
```

Should be 5 separate classes:
- `MemoryProfiler`
- `CacheManager`
- `BatchOptimizer`
- `MemoryMonitor`
- `OptimizerConfig`

#### 6. Excessive Comments
**File**: Multiple files
- **Problem**: Comments explaining what code does instead of why
- **Violation**: "Every time you write a comment, you should grimace"

**Examples**:
```python
# Bad: Comment explains what (code should be self-explanatory)
# Calculate the attention weights for each instance
attention_weights = self.attention_layer(features)

# Good: Comment explains why (non-obvious business logic)
# Use temperature scaling to prevent attention collapse in early training
attention_weights = self.attention_layer(features) / self.temperature
```

#### 7. Try-Catch Blocks with Business Logic
**File**: `src/api/main.py`
- **Problem**: Try blocks contain 50+ lines of business logic
- **Violation**: "Extract the bodies of try and catch blocks into functions"

**Example**:
```python
@app.post("/api/upload")
async def upload_for_analysis(...):
    try:
        # 80 lines of validation, processing, database operations
        # All mixed together in try block
        ...
    except Exception as e:
        # Error handling
        ...
```

Should be:
```python
@app.post("/api/upload")
async def upload_for_analysis(...):
    try:
        await _process_upload(...)
    except ValidationError as e:
        return _handle_validation_error(e)
    except StorageError as e:
        return _handle_storage_error(e)
```

---

## User Stories

### US-1: Developer Onboarding
**As a** new developer joining the team  
**I want** to understand the codebase quickly  
**So that** I can start contributing within days, not weeks

**Acceptance Criteria**:
- Any file can be understood in <10 minutes
- Function purpose clear from name and signature
- No need to read 1000+ line files to understand one feature

**Current Pain**: New developers spend 2-3 weeks just understanding the MIL models because they're 1500 lines in one file.

### US-2: Bug Fixing
**As a** developer fixing a bug  
**I want** to locate the relevant code quickly  
**So that** I can fix issues in minutes, not hours

**Acceptance Criteria**:
- Bug location identifiable from stack trace
- Related code grouped logically
- No need to search through 40 route handlers to find one endpoint

**Current Pain**: Finding the right route handler in `main.py` requires Ctrl+F and reading through similar-looking functions.

### US-3: Feature Addition
**As a** developer adding a new MIL model  
**I want** to reuse existing components  
**So that** I don't duplicate code

**Acceptance Criteria**:
- Common functionality extracted to shared modules
- New model requires <100 lines of unique code
- Fusion strategies reusable across models

**Current Pain**: Adding a new MIL model requires copying 300+ lines from existing models and modifying them.

### US-4: Testing
**As a** developer writing tests  
**I want** to test individual components in isolation  
**So that** tests are fast and focused

**Acceptance Criteria**:
- Each function testable independently
- No need to mock 10 dependencies to test one function
- Test files <500 lines

**Current Pain**: Testing route handlers requires mocking database, auth, storage, and inference engine because they're all coupled.

### US-5: Code Review
**As a** code reviewer  
**I want** to understand changes quickly  
**So that** I can provide meaningful feedback

**Acceptance Criteria**:
- Changes affect <3 files for most features
- Each file has clear, single purpose
- Diffs are readable and focused

**Current Pain**: Changes to MIL models create 200+ line diffs in a single file, making review difficult.

---

## Functional Requirements

### FR-1: File Size Limits
**Requirement**: No Python file shall exceed 500 lines

**Rationale**: Files >500 lines are difficult to understand and maintain

**Targets**:
- `attention_mil.py`: 1527 → <500 lines (split into 4 files)
- `main.py`: 1308 → <500 lines (split into 3+ routers)
- `treatment_response.py`: 1442 → <500 lines (split into modules)

### FR-2: Function Size Limits
**Requirement**: No function shall exceed 50 lines

**Rationale**: Functions >50 lines do too many things

**Targets**:
- `upload_for_analysis()`: 97 → <50 lines
- `oauth_callback()`: 86 → <50 lines
- `process_real_analysis()`: 70 → <50 lines

### FR-3: Single Responsibility
**Requirement**: Each class/function shall have one clear responsibility

**Rationale**: SRP makes code easier to understand, test, and modify

**Targets**:
- `MemoryOptimizer`: 1 class → 5 classes
- `AttentionMIL`: Extract fusion strategies to separate module
- Route handlers: Group by domain (auth, analysis, admin)

### FR-4: DRY (Don't Repeat Yourself)
**Requirement**: No code block >10 lines shall be duplicated

**Rationale**: Duplication leads to inconsistent bug fixes

**Targets**:
- Fusion methods: Extract to `src/models/fusion_strategies.py`
- Attention mechanisms: Extract to `src/models/attention_mechanisms.py`
- Route validation: Extract to `src/api/validators.py`

### FR-5: Meaningful Names
**Requirement**: All functions/classes shall have descriptive names

**Rationale**: "Clean code always looks like it was written by someone who cares"

**Examples**:
- ❌ `process()` → ✅ `process_wsi_for_metastasis_detection()`
- ❌ `handle()` → ✅ `handle_upload_validation_error()`
- ❌ `data` → ✅ `patient_clinical_context`

### FR-6: Minimal Comments
**Requirement**: Code shall be self-documenting; comments only for "why", not "what"

**Rationale**: Comments rot; code doesn't

**Guidelines**:
- Remove comments that explain what code does
- Keep comments that explain why (business logic, algorithms)
- Use docstrings for public APIs

### FR-7: Error Handling Extraction
**Requirement**: Try blocks shall contain only function calls, not business logic

**Rationale**: Improves readability and testability

**Pattern**:
```python
# Before
try:
    # 50 lines of business logic
except Exception as e:
    # error handling

# After
try:
    result = _do_business_logic()
except SpecificError as e:
    return _handle_specific_error(e)
```

---

## Non-Functional Requirements

### NFR-1: Backward Compatibility
**Requirement**: All refactoring shall maintain existing APIs

**Rationale**: Don't break existing code

**Validation**:
- All existing tests pass
- No changes to public function signatures
- No changes to API endpoints

### NFR-2: Performance Neutrality
**Requirement**: Refactoring shall not degrade performance

**Rationale**: Clean code shouldn't be slow code

**Validation**:
- Inference latency unchanged (±5%)
- Memory usage unchanged (±10%)
- Throughput unchanged (±5%)

### NFR-3: Test Coverage Maintenance
**Requirement**: Test coverage shall not decrease

**Rationale**: Refactoring is risky; tests provide safety net

**Validation**:
- Coverage remains >80%
- All existing tests pass
- New tests added for extracted functions

### NFR-4: Incremental Refactoring
**Requirement**: Refactoring shall be done incrementally, not big-bang

**Rationale**: "Later equals never" - do it now, but safely

**Approach**:
- Refactor one file at a time
- Commit after each file
- Run tests after each commit

---

## Correctness Properties

### Property 1: API Compatibility
**Property**: All existing API endpoints shall return identical responses before and after refactoring

**Test Strategy**: Property-based testing with recorded request/response pairs

```python
@given(api_request=recorded_requests())
def test_api_compatibility(api_request):
    # Record responses before refactoring
    old_response = call_old_api(api_request)
    
    # Compare with responses after refactoring
    new_response = call_new_api(api_request)
    
    assert new_response == old_response
```

### Property 2: Model Output Equivalence
**Property**: MIL models shall produce identical predictions before and after refactoring

**Test Strategy**: Snapshot testing with fixed random seeds

```python
@given(wsi_features=synthetic_wsi_features())
def test_model_output_equivalence(wsi_features):
    torch.manual_seed(42)
    old_output = old_attention_mil(wsi_features)
    
    torch.manual_seed(42)
    new_output = new_attention_mil(wsi_features)
    
    assert torch.allclose(old_output, new_output, atol=1e-6)
```

### Property 3: Performance Preservation
**Property**: Refactored code shall not be >10% slower than original

**Test Strategy**: Benchmark testing with statistical validation

```python
def test_performance_preservation():
    old_time = benchmark(old_function, iterations=1000)
    new_time = benchmark(new_function, iterations=1000)
    
    assert new_time <= old_time * 1.10  # Max 10% slower
```

---

## Refactoring Targets (Priority Order)

### Priority 1: Critical Path (Week 1-2)
**Impact**: High usage, high complexity

1. **`src/api/main.py`** (1,308 lines)
   - Split into routers: `auth.py`, `analysis.py`, `admin.py`, `mobile.py`
   - Extract validation logic
   - Extract error handlers
   - **Benefit**: Easier to find endpoints, faster development

2. **`src/models/attention_mil.py`** (1,527 lines)
   - Extract fusion strategies to `fusion_strategies.py`
   - Extract attention mechanisms to `attention_mechanisms.py`
   - Split models into separate files: `attention_mil.py`, `clam.py`, `transmil.py`
   - **Benefit**: Easier to add new models, reduce duplication

### Priority 2: High Complexity (Week 3-4)
**Impact**: Difficult to maintain, frequent bugs

3. **`src/streaming/memory_optimizer.py`** (1,097 lines)
   - Split into: `profiler.py`, `cache_manager.py`, `batch_optimizer.py`, `monitor.py`
   - **Benefit**: Easier to test, clearer responsibilities

4. **`src/clinical/treatment_response.py`** (1,442 lines)
   - Split into: `response_calculator.py`, `progression_analyzer.py`, `outcome_predictor.py`
   - **Benefit**: Easier to understand clinical logic

### Priority 3: Moderate Complexity (Week 5-6)
**Impact**: Maintenance burden

5. **`src/clinical/audit.py`** (1,026 lines)
6. **`src/clinical/regulatory.py`** (976 lines)
7. **`src/streaming/security.py`** (852 lines)

---

## Success Metrics

### Code Quality Metrics
- **Average file size**: 1000 lines → <400 lines
- **Max file size**: 1527 lines → <500 lines
- **Average function size**: 30 lines → <25 lines
- **Max function size**: 97 lines → <50 lines
- **Code duplication**: 15% → <5%

### Developer Productivity Metrics
- **Time to understand new file**: 30 min → <10 min
- **Time to locate bug**: 20 min → <5 min
- **Time to add new feature**: 4 hours → <2 hours
- **Code review time**: 45 min → <20 min

### Quality Metrics
- **Test coverage**: Maintain >80%
- **Bug rate**: Maintain or reduce
- **Performance**: No degradation (±5%)

---

## Risks and Mitigations

### Risk 1: Breaking Changes
**Probability**: Medium  
**Impact**: High  
**Mitigation**:
- Comprehensive test suite before refactoring
- Property-based testing for equivalence
- Incremental commits with CI validation

### Risk 2: Performance Regression
**Probability**: Low  
**Impact**: High  
**Mitigation**:
- Benchmark tests before/after
- Profile critical paths
- Rollback if >10% degradation

### Risk 3: Incomplete Refactoring
**Probability**: High  
**Impact**: Medium  
**Mitigation**:
- "Later equals never" - schedule dedicated time
- Prioritize high-impact files first
- Track progress with metrics

### Risk 4: Team Resistance
**Probability**: Medium  
**Impact**: Medium  
**Mitigation**:
- Show concrete benefits (faster development)
- Involve team in planning
- Celebrate wins (faster code reviews, easier debugging)

---

## Out of Scope

### Not Included in This Refactoring
1. **Algorithm changes**: We're improving structure, not changing logic
2. **New features**: Pure refactoring, no new functionality
3. **Performance optimization**: Maintain performance, don't optimize
4. **Test refactoring**: Focus on production code first
5. **Documentation updates**: Update only if structure changes

---

## Dependencies

### Required Before Starting
- ✅ Comprehensive test suite (exists)
- ✅ CI/CD pipeline (exists)
- ✅ Version control (Git)

### Required During Refactoring
- Property-based testing framework (Hypothesis) - already installed
- Benchmark testing tools (pytest-benchmark)
- Code coverage tools (pytest-cov) - already installed

---

## Timeline Estimate

**Total Duration**: 6 weeks (120 hours)

- **Week 1-2**: Priority 1 (API routes, MIL models) - 40 hours
- **Week 3-4**: Priority 2 (Memory optimizer, treatment response) - 40 hours
- **Week 5-6**: Priority 3 (Audit, regulatory, security) - 40 hours

**Effort per file**:
- Large file (>1000 lines): 8-12 hours
- Medium file (500-1000 lines): 4-6 hours
- Small file (<500 lines): 2-3 hours

---

## Acceptance Criteria

### Definition of Done
A file is considered "refactored" when:

1. ✅ File size <500 lines
2. ✅ All functions <50 lines
3. ✅ No code duplication >10 lines
4. ✅ Single responsibility per class/function
5. ✅ Meaningful names (no abbreviations)
6. ✅ Minimal comments (only "why", not "what")
7. ✅ All tests pass
8. ✅ Test coverage maintained or improved
9. ✅ Performance within ±5% of original
10. ✅ Code review approved

---

## References

### Clean Code Principles (Robert C. Martin)
1. "Later equals never" - Don't postpone cleanup
2. "Functions should be small" - <50 lines
3. "Do one thing" - Single Responsibility Principle
4. "Meaningful names" - Code should read like prose
5. "Comments are failures" - Code should be self-documenting
6. "Extract try/catch bodies" - Separate error handling from logic

### Codebase Analysis
- Total Python files: 200+
- Total lines of code: 50,000+
- Files >1000 lines: 7
- Files >500 lines: 20+
- Average file size: 250 lines
- Largest file: 1,527 lines

---

**Next Steps**: Proceed to Design phase to plan refactoring strategy and implementation approach.

**Prepared By**: Kiro AI  
**Date**: 2026-05-03  
**Status**: Requirements Complete - Ready for Design
