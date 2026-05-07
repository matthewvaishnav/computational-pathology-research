# Clean Code Refactoring - Design

**Feature**: clean-code-refactoring  
**Date**: 2026-05-03  
**Status**: Design Phase

---

## Architecture

### Refactoring Strategy: Strangler Fig Pattern

No big-bang rewrite. Extract → Test → Replace → Delete old.

```
Old Code (1500 lines)
    ↓
Extract Module A (200 lines) → Test → Import in old code
    ↓
Extract Module B (200 lines) → Test → Import in old code
    ↓
Extract Module C (200 lines) → Test → Import in old code
    ↓
Old code now thin wrapper → Delete wrapper → Done
```

**Why**: Safe. Incremental. Rollback easy. Tests run every step.

---

## Priority 1: API Routes (`src/api/main.py`)

### Current: 1308 lines, 40+ routes mixed

**Problem**:
```python
# main.py - everything in one file
@app.post("/api/register")  # Auth
@app.post("/api/upload")    # Analysis
@app.get("/api/users")      # Admin
@app.post("/api/mobile/sync")  # Mobile
# ... 36 more routes
```

### Target: Router pattern

**New structure**:
```
src/api/
├── main.py (100 lines - app setup only)
├── routers/
│   ├── auth.py (200 lines - register, login, oauth)
│   ├── analysis.py (250 lines - upload, results, dicom)
│   ├── admin.py (150 lines - users, config, audit)
│   ├── mobile.py (100 lines - sync, offline, models)
│   └── monitoring.py (150 lines - health, metrics, ids)
├── dependencies.py (100 lines - get_db, get_user, get_engine)
├── validators.py (150 lines - file validation, input validation)
└── errors.py (100 lines - error handlers)
```

**Implementation**:

Step 1: Extract dependencies
```python
# src/api/dependencies.py
from typing import Generator
from sqlalchemy.orm import Session

def get_db_session() -> Generator[Session, None, None]:
    """DB session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_inference_engine() -> InferenceEngine:
    """Inference engine singleton."""
    return InferenceEngine.get_instance()

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Extract user from JWT."""
    return verify_jwt_token(token)
```

Step 2: Extract auth router
```python
# src/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_db_session
from ..validators import validate_email, validate_password

router = APIRouter(prefix="/api", tags=["auth"])

@router.post("/register")
async def register_user(user_data: UserRegistration, db: Session = Depends(get_db_session)):
    """Register new user."""
    validate_email(user_data.email)
    validate_password(user_data.password)
    
    if _user_exists(db, user_data.email):
        raise HTTPException(409, "User exists")
    
    user = _create_user(db, user_data)
    return {"user_id": user.id}

def _user_exists(db: Session, email: str) -> bool:
    """Check if user exists."""
    return db.query(User).filter(User.email == email).first() is not None

def _create_user(db: Session, user_data: UserRegistration) -> User:
    """Create user in DB."""
    hashed_pw = hash_password(user_data.password)
    user = User(email=user_data.email, password_hash=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
```

Step 3: Extract validators
```python
# src/api/validators.py
import re
from fastapi import HTTPException

def validate_email(email: str) -> None:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise HTTPException(400, "Invalid email")

def validate_password(password: str) -> None:
    """Validate password strength."""
    if len(password) < 8:
        raise HTTPException(400, "Password too short")
    if not any(c.isupper() for c in password):
        raise HTTPException(400, "Password needs uppercase")
    if not any(c.isdigit() for c in password):
        raise HTTPException(400, "Password needs digit")

def validate_file_upload(file: UploadFile, max_size_mb: int = 100) -> None:
    """Validate uploaded file."""
    allowed_types = {".svs", ".tif", ".tiff", ".ndpi", ".dcm"}
    
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(400, f"File type {ext} not allowed")
    
    if file.size > max_size_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {max_size_mb}MB")
```

Step 4: Wire routers in main
```python
# src/api/main.py (now 100 lines)
from fastapi import FastAPI
from .routers import auth, analysis, admin, mobile, monitoring

app = FastAPI(title="HistoCore API")

# Include routers
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(admin.router)
app.include_router(mobile.router)
app.include_router(monitoring.router)

@app.on_event("startup")
async def startup():
    """Initialize services."""
    await init_database()
    await init_inference_engine()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Migration path**:
1. Create `dependencies.py` - extract 3 dependency functions
2. Create `validators.py` - extract validation logic
3. Create `routers/auth.py` - move auth routes
4. Test auth routes work
5. Create `routers/analysis.py` - move analysis routes
6. Test analysis routes work
7. Repeat for admin, mobile, monitoring
8. Slim down `main.py` to app setup only
9. Delete old route code

**Tests**: All existing API tests pass. No new tests needed (behavior unchanged).

---

## Priority 2: MIL Models (`src/models/attention_mil.py`)

### Current: 1527 lines, 4 classes, massive duplication

**Problem**:
```python
# attention_mil.py - 1527 lines
class AttentionMIL:  # 416 lines
    def _early_fusion(self, ...):  # 65 lines
        # Fusion logic
    def _late_fusion(self, ...):  # 66 lines
        # Fusion logic

class CLAM:  # 587 lines
    def _early_fusion_clam(self, ...):  # 95 lines
        # Same fusion logic, slightly different
    def _late_fusion_clam(self, ...):  # 97 lines
        # Same fusion logic, slightly different

class TransMIL:  # 504 lines
    def _early_fusion_transmil(self, ...):  # 79 lines
        # Same fusion logic again
    def _late_fusion_transmil(self, ...):  # 83 lines
        # Same fusion logic again
```

**Duplication**: 200+ lines of fusion code repeated 3x.

### Target: Extract common components

**New structure**:
```
src/models/
├── attention_mil.py (300 lines - AttentionMIL class only)
├── clam.py (350 lines - CLAM class only)
├── transmil.py (350 lines - TransMIL class only)
├── mil_base.py (150 lines - base class, common methods)
├── fusion_strategies.py (200 lines - early/late fusion)
├── attention_mechanisms.py (150 lines - attention computation)
└── factory.py (50 lines - create_attention_model)
```

**Implementation**:

Step 1: Extract fusion strategies
```python
# src/models/fusion_strategies.py
import torch
import torch.nn as nn
from typing import Dict, Optional

class FusionStrategy(nn.Module):
    """Base class for multimodal fusion."""
    
    def forward(self, features: torch.Tensor, multimodal_data: Optional[Dict] = None) -> torch.Tensor:
        raise NotImplementedError

class EarlyFusion(FusionStrategy):
    """Concatenate features before model."""
    
    def __init__(self, feature_dim: int, modality_dims: Dict[str, int], output_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })
        total_dim = output_dim * (1 + len(modality_dims))
        self.fusion = nn.Linear(total_dim, output_dim)
    
    def forward(self, features: torch.Tensor, multimodal_data: Optional[Dict] = None) -> torch.Tensor:
        if multimodal_data is None:
            return features
        
        # Project each modality
        projected = [features]
        for name, data in multimodal_data.items():
            proj = self.projections[name](data)
            projected.append(proj)
        
        # Concatenate and fuse
        concat = torch.cat(projected, dim=-1)
        return self.fusion(concat)

class LateFusion(FusionStrategy):
    """Combine predictions after model."""
    
    def __init__(self, num_classes: int, modality_dims: Dict[str, int]):
        super().__init__()
        self.modality_heads = nn.ModuleDict({
            name: nn.Linear(dim, num_classes)
            for name, dim in modality_dims.items()
        })
        self.fusion_weights = nn.Parameter(torch.ones(1 + len(modality_dims)))
    
    def forward(self, logits: torch.Tensor, multimodal_features: Optional[Dict] = None) -> torch.Tensor:
        if multimodal_features is None:
            return logits
        
        # Get predictions from each modality
        all_logits = [logits]
        for name, features in multimodal_features.items():
            modal_logits = self.modality_heads[name](features)
            all_logits.append(modal_logits)
        
        # Weighted fusion
        stacked = torch.stack(all_logits, dim=0)
        weights = torch.softmax(self.fusion_weights, dim=0)
        return torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
```

Step 2: Extract attention mechanisms
```python
# src/models/attention_mechanisms.py
import torch
import torch.nn as nn

class AttentionMechanism(nn.Module):
    """Base class for attention."""
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class GatedAttention(AttentionMechanism):
    """Gated attention from CLAM."""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.attention_V = nn.Linear(feature_dim, hidden_dim)
        self.attention_U = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, instances, feature_dim]
        V = torch.tanh(self.attention_V(features))
        U = torch.sigmoid(self.attention_U(features))
        attention_scores = self.attention_w(V * U)  # Gating
        return torch.softmax(attention_scores, dim=1)

class TransformerAttention(AttentionMechanism):
    """Multi-head self-attention from TransMIL."""
    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, instances, feature_dim]
        attn_output, attn_weights = self.mha(features, features, features)
        return self.norm(features + attn_output), attn_weights
```

Step 3: Create base class
```python
# src/models/mil_base.py
import torch
import torch.nn as nn
from typing import Optional, Dict
from .fusion_strategies import FusionStrategy
from .attention_mechanisms import AttentionMechanism

class MILBase(nn.Module):
    """Base class for MIL models."""
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        attention: AttentionMechanism,
        fusion: Optional[FusionStrategy] = None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.attention = attention
        self.fusion = fusion
    
    def compute_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention weights."""
        return self.attention(features)
    
    def aggregate_features(self, features: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """Aggregate features using attention."""
        return torch.sum(features * attention_weights, dim=1)
    
    def apply_fusion(self, features: torch.Tensor, multimodal_data: Optional[Dict] = None) -> torch.Tensor:
        """Apply multimodal fusion if available."""
        if self.fusion is None or multimodal_data is None:
            return features
        return self.fusion(features, multimodal_data)
```

Step 4: Refactor AttentionMIL
```python
# src/models/attention_mil.py (now 300 lines)
import torch
import torch.nn as nn
from .mil_base import MILBase
from .attention_mechanisms import GatedAttention
from .fusion_strategies import EarlyFusion, LateFusion

class AttentionMIL(MILBase):
    """Attention-based MIL model."""
    
    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        multimodal_config: Optional[Dict] = None,
        fusion_type: str = "none"
    ):
        # Setup fusion
        fusion = self._create_fusion(fusion_type, feature_dim, num_classes, multimodal_config)
        
        # Setup attention
        attention = GatedAttention(feature_dim, hidden_dim)
        
        super().__init__(feature_dim, num_classes, attention, fusion)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def _create_fusion(self, fusion_type: str, feature_dim: int, num_classes: int, config: Optional[Dict]) -> Optional[FusionStrategy]:
        """Create fusion strategy."""
        if fusion_type == "none" or config is None:
            return None
        elif fusion_type == "early":
            return EarlyFusion(feature_dim, config["modality_dims"], feature_dim)
        elif fusion_type == "late":
            return LateFusion(num_classes, config["modality_dims"])
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, features: torch.Tensor, multimodal_data: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        # Apply early fusion if configured
        if self.fusion and isinstance(self.fusion, EarlyFusion):
            features = self.apply_fusion(features, multimodal_data)
        
        # Extract features
        h = self.feature_extractor(features)
        
        # Compute attention
        attention_weights = self.compute_attention(h)
        
        # Aggregate
        aggregated = self.aggregate_features(h, attention_weights)
        
        # Classify
        logits = self.classifier(aggregated)
        
        # Apply late fusion if configured
        if self.fusion and isinstance(self.fusion, LateFusion):
            logits = self.fusion(logits, multimodal_data)
        
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "aggregated_features": aggregated
        }
```

**Migration path**:
1. Create `fusion_strategies.py` - extract EarlyFusion, LateFusion
2. Create `attention_mechanisms.py` - extract GatedAttention, TransformerAttention
3. Create `mil_base.py` - extract common base class
4. Refactor `AttentionMIL` to use extracted components
5. Test AttentionMIL produces identical outputs (property test)
6. Refactor `CLAM` to use extracted components
7. Test CLAM produces identical outputs
8. Refactor `TransMIL` to use extracted components
9. Test TransMIL produces identical outputs
10. Delete old duplicated code

**Tests**: Property-based test for output equivalence.

```python
@given(features=synthetic_wsi_features())
def test_attention_mil_refactor_equivalence(features):
    """Refactored model produces identical outputs."""
    torch.manual_seed(42)
    old_model = OldAttentionMIL()
    old_output = old_model(features)
    
    torch.manual_seed(42)
    new_model = AttentionMIL()
    new_output = new_model(features)
    
    assert torch.allclose(old_output["logits"], new_output["logits"], atol=1e-6)
```

---

## Priority 3: Memory Optimizer (`src/streaming/memory_optimizer.py`)

### Current: 1097 lines, God object

**Problem**: One class does 5 jobs.

```python
class MemoryOptimizer:
    # Memory profiling (100 lines)
    def profile_memory_usage(self): ...
    def track_allocations(self): ...
    
    # Cache management (150 lines)
    def manage_cache(self): ...
    def evict_cache_entries(self): ...
    
    # Optimization (200 lines)
    def optimize_batch_size(self): ...
    def optimize_tile_size(self): ...
    
    # Monitoring (150 lines)
    def monitor_memory(self): ...
    def send_alerts(self): ...
    
    # Config (100 lines)
    def load_config(self): ...
```

### Target: 5 focused classes

**New structure**:
```
src/streaming/memory/
├── profiler.py (200 lines - MemoryProfiler)
├── cache_manager.py (250 lines - CacheManager)
├── batch_optimizer.py (300 lines - BatchOptimizer)
├── monitor.py (200 lines - MemoryMonitor)
├── config.py (150 lines - OptimizerConfig)
└── coordinator.py (100 lines - MemoryCoordinator facade)
```

**Implementation**:

```python
# src/streaming/memory/profiler.py
import psutil
import torch
from dataclasses import dataclass
from typing import Dict

@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    cpu_used_mb: float
    cpu_available_mb: float
    gpu_used_mb: float
    gpu_available_mb: float

class MemoryProfiler:
    """Profile memory usage."""
    
    def __init__(self):
        self.snapshots = []
    
    def take_snapshot(self) -> MemorySnapshot:
        """Capture current memory state."""
        cpu_mem = psutil.virtual_memory()
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.mem_get_info()
            gpu_used = (gpu_mem[1] - gpu_mem[0]) / 1024**2
            gpu_avail = gpu_mem[0] / 1024**2
        else:
            gpu_used = gpu_avail = 0.0
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_used_mb=cpu_mem.used / 1024**2,
            cpu_available_mb=cpu_mem.available / 1024**2,
            gpu_used_mb=gpu_used,
            gpu_available_mb=gpu_avail
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not self.snapshots:
            return {"cpu_mb": 0.0, "gpu_mb": 0.0}
        
        return {
            "cpu_mb": max(s.cpu_used_mb for s in self.snapshots),
            "gpu_mb": max(s.gpu_used_mb for s in self.snapshots)
        }
```

```python
# src/streaming/memory/cache_manager.py
from collections import OrderedDict
from typing import Any, Optional

class CacheManager:
    """LRU cache for tiles/features."""
    
    def __init__(self, max_size_mb: int = 1000):
        self.max_size_mb = max_size_mb
        self.cache = OrderedDict()
        self.current_size_mb = 0.0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any, size_mb: float) -> None:
        """Add item to cache."""
        # Evict if needed
        while self.current_size_mb + size_mb > self.max_size_mb and self.cache:
            self._evict_lru()
        
        self.cache[key] = value
        self.current_size_mb += size_mb
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        key, value = self.cache.popitem(last=False)
        size = self._estimate_size(value)
        self.current_size_mb -= size
```

```python
# src/streaming/memory/coordinator.py
from .profiler import MemoryProfiler
from .cache_manager import CacheManager
from .batch_optimizer import BatchOptimizer
from .monitor import MemoryMonitor

class MemoryCoordinator:
    """Facade for memory management."""
    
    def __init__(self, config: OptimizerConfig):
        self.profiler = MemoryProfiler()
        self.cache = CacheManager(config.cache_size_mb)
        self.optimizer = BatchOptimizer(config)
        self.monitor = MemoryMonitor(config.alert_threshold_mb)
    
    def optimize_for_workload(self, workload_size: int) -> Dict[str, int]:
        """Optimize batch/tile sizes for workload."""
        snapshot = self.profiler.take_snapshot()
        self.monitor.check_usage(snapshot)
        return self.optimizer.compute_optimal_sizes(snapshot, workload_size)
```

**Migration path**:
1. Create `memory/profiler.py` - extract profiling logic
2. Create `memory/cache_manager.py` - extract cache logic
3. Create `memory/batch_optimizer.py` - extract optimization logic
4. Create `memory/monitor.py` - extract monitoring logic
5. Create `memory/config.py` - extract config logic
6. Create `memory/coordinator.py` - facade for backward compat
7. Update imports in streaming code
8. Test all memory operations work
9. Delete old `memory_optimizer.py`

---

## Testing Strategy

### Property-Based Tests

**Property 1: API Response Equivalence**
```python
@given(request=recorded_api_requests())
def test_api_refactor_equivalence(request):
    """API returns same responses after refactor."""
    old_response = call_old_api(request)
    new_response = call_new_api(request)
    assert old_response == new_response
```

**Property 2: Model Output Equivalence**
```python
@given(features=synthetic_features())
def test_model_refactor_equivalence(features):
    """Models produce identical outputs after refactor."""
    torch.manual_seed(42)
    old_output = old_model(features)
    
    torch.manual_seed(42)
    new_output = new_model(features)
    
    assert torch.allclose(old_output, new_output, atol=1e-6)
```

**Property 3: Performance Preservation**
```python
def test_performance_no_regression():
    """Refactored code not >10% slower."""
    old_time = benchmark(old_function, n=1000)
    new_time = benchmark(new_function, n=1000)
    assert new_time <= old_time * 1.10
```

### Unit Tests

Each extracted function gets unit test.

```python
def test_validate_email():
    """Email validation works."""
    validate_email("user@example.com")  # OK
    
    with pytest.raises(HTTPException):
        validate_email("invalid")  # Fails

def test_early_fusion():
    """Early fusion combines modalities."""
    fusion = EarlyFusion(1024, {"clinical": 128}, 1024)
    features = torch.randn(1, 100, 1024)
    clinical = torch.randn(1, 128)
    
    fused = fusion(features, {"clinical": clinical})
    assert fused.shape == (1, 100, 1024)
```

---

## Rollback Strategy

Each refactor step committed separately. Rollback = `git revert`.

**Commit pattern**:
```
refactor(api): extract auth router
refactor(api): extract analysis router
refactor(api): extract validators
refactor(models): extract fusion strategies
refactor(models): extract attention mechanisms
refactor(models): refactor AttentionMIL to use extracted components
```

If tests fail after commit → revert immediately.

---

## Performance Validation

Benchmark before/after each refactor.

```python
# benchmark.py
import time
import torch

def benchmark_inference(model, features, n=1000):
    """Benchmark model inference."""
    times = []
    for _ in range(n):
        start = time.perf_counter()
        with torch.no_grad():
            model(features)
        times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": np.mean(times) * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000
    }

# Run before refactor
old_stats = benchmark_inference(old_model, test_features)

# Run after refactor
new_stats = benchmark_inference(new_model, test_features)

# Validate <10% regression
assert new_stats["mean_ms"] <= old_stats["mean_ms"] * 1.10
```

---

## Code Quality Metrics

Track before/after:

```python
# metrics.py
def compute_metrics(directory: str) -> Dict:
    """Compute code quality metrics."""
    files = list(Path(directory).rglob("*.py"))
    
    total_lines = 0
    total_functions = 0
    long_functions = 0  # >50 lines
    long_files = 0  # >500 lines
    
    for file in files:
        lines = file.read_text().splitlines()
        total_lines += len(lines)
        
        if len(lines) > 500:
            long_files += 1
        
        # Count functions
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                func_lines = node.end_lineno - node.lineno
                if func_lines > 50:
                    long_functions += 1
    
    return {
        "total_lines": total_lines,
        "avg_file_size": total_lines / len(files),
        "total_functions": total_functions,
        "avg_function_size": total_lines / total_functions,
        "long_files": long_files,
        "long_functions": long_functions
    }
```

**Target metrics**:
- Avg file size: 1000 → <400 lines
- Max file size: 1527 → <500 lines
- Long functions (>50 lines): 20 → 0
- Long files (>500 lines): 20 → 0

---

## Timeline

**Week 1-2: API Routes**
- Day 1: Extract dependencies, validators
- Day 2: Extract auth router
- Day 3: Extract analysis router
- Day 4: Extract admin router
- Day 5: Extract mobile, monitoring routers
- Day 6-7: Test, benchmark, commit

**Week 3-4: MIL Models**
- Day 1: Extract fusion strategies
- Day 2: Extract attention mechanisms
- Day 3: Create base class
- Day 4: Refactor AttentionMIL
- Day 5: Refactor CLAM
- Day 6: Refactor TransMIL
- Day 7: Test, benchmark, commit

**Week 5-6: Memory Optimizer**
- Day 1: Extract profiler
- Day 2: Extract cache manager
- Day 3: Extract batch optimizer
- Day 4: Extract monitor
- Day 5: Create coordinator
- Day 6-7: Test, benchmark, commit

---

## Success Criteria

Refactor complete when:

1. ✅ All files <500 lines
2. ✅ All functions <50 lines
3. ✅ No code duplication >10 lines
4. ✅ All tests pass
5. ✅ Test coverage maintained (>80%)
6. ✅ Performance within ±5%
7. ✅ Code review approved

---

**Next**: Create tasks.md with step-by-step implementation plan.

**Date**: 2026-05-03  
**Status**: Design Complete
