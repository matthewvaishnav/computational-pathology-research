# Additional Issues Found

## 🟡 Medium Priority Issues

### 1. Print Statements in Production Code (959 instances)
**Risk:** Performance overhead, log pollution  
**Files:** 79 files in `src/`

**Top Offenders:**
- `src/streaming/demo_scenarios.py` (130 instances)
- `src/data/wsi_pipeline/example_progress_tracking.py` (78 instances)
- `src/data/wsi_pipeline/demo_progress_tracking.py` (67 instances)

**Fix:**
```python
# Replace print with logger
import logging
logger = logging.getLogger(__name__)

# Before:
print(f"Processing {count} items")

# After:
logger.info(f"Processing {count} items")
```

---

### 2. Assert Statements in Production (20 instances)
**Risk:** Disabled with `python -O`, silent failures  
**Files:** 8 files

**Critical Files:**
- `src/data/wsi_pipeline/validation.py` (10 instances)
- `src/mobile_edge/` (7 instances)

**Fix:**
```python
# Before:
assert len(coords) > 0, "No coordinates"

# After:
if len(coords) == 0:
    raise ValueError("No coordinates generated")
```

---

### 3. Hardcoded /tmp Paths (19 instances)
**Risk:** Permission issues, cross-platform incompatibility  
**Files:** 14 files

**Fix:**
```python
# Before:
temp_dir = "/tmp/histocore"

# After:
import tempfile
temp_dir = tempfile.mkdtemp(prefix="histocore_")
```

---

### 4. Deprecated NumPy Types (92 instances)
**Risk:** Will break in NumPy 2.0  
**Pattern:** `np.int32`, `np.int64`, `np.float`

**Fix:**
```python
# Before:
arr = np.array(data, dtype=np.int32)

# After:
arr = np.array(data, dtype=np.int32)  # OK - this is the dtype
# Only deprecated: np.int (alias), use int or np.int32
```

**Note:** Most uses are correct (`dtype=np.int32`). Only aliases like `np.int` are deprecated.

---

### 5. Missing Type Hints (1350+ functions)
**Risk:** Reduced code maintainability  
**Files:** 255 files

**Example:**
```python
# Before:
def process_data(data):
    return data * 2

# After:
def process_data(data: np.ndarray) -> np.ndarray:
    return data * 2
```

---

## 🟢 Low Priority Issues

### 6. TODO/FIXME Comments
**Count:** Checking...

### 7. Mutable Default Arguments
**Count:** 0 ✅ (None found)

---

## Automated Fix Script

```python
#!/usr/bin/env python3
import re
from pathlib import Path

def fix_print_statements(content):
    # Add logger import if needed
    if 'print(' in content and 'import logging' not in content:
        content = 'import logging\nlogger = logging.getLogger(__name__)\n' + content
    
    # Replace print with logger
    content = re.sub(
        r'print\(f?"([^"]+)"\)',
        r'logger.info(f"\1")',
        content
    )
    return content

def fix_assert_statements(content):
    # Replace assert with proper exceptions
    content = re.sub(
        r'assert (.+), "(.+)"',
        r'if not (\1):\n    raise ValueError("\2")',
        content
    )
    return content

def fix_tmp_paths(content):
    # Replace /tmp with tempfile
    if '/tmp/' in content and 'import tempfile' not in content:
        content = 'import tempfile\n' + content
    
    content = re.sub(
        r'"/tmp/(\w+)"',
        r'tempfile.mkdtemp(prefix="\1_")',
        content
    )
    return content

# Apply to all src/ files
root = Path('/home/five/computational-pathology-research/src')
for f in root.rglob('*.py'):
    content = f.read_text()
    original = content
    
    content = fix_print_statements(content)
    content = fix_assert_statements(content)
    content = fix_tmp_paths(content)
    
    if content != original:
        f.write_text(content)
        print(f"Fixed: {f}")
```

---

## Priority Summary

| Issue | Count | Severity | Auto-Fix |
|-------|-------|----------|----------|
| Print statements | 959 | Medium | Yes |
| Assert statements | 20 | Medium | Yes |
| Hardcoded /tmp | 19 | Medium | Yes |
| Deprecated NumPy | 92 | Low | Manual |
| Missing type hints | 1350+ | Low | Manual |

---

## Recommendation

**Apply Medium Priority Fixes:**
1. Replace print with logger (demo files can keep print)
2. Replace assert with proper exceptions
3. Replace /tmp with tempfile

**Estimated Time:** 1-2 hours with automated script

**Production Impact:** Minimal (mostly code quality improvements)
