#!/usr/bin/env python3
"""
Update Documentation Metrics Script
Updates all documentation files with current HistoCore metrics
"""

import re
from pathlib import Path
from typing import List, Tuple

# Current metrics
CURRENT_METRICS = {
    "test_count": "3,171",
    "test_count_no_comma": "3171",
    "coverage": "55%",
    "optimization": "8-12x",
    "validation_auc": "100%",
    "test_accuracy": "85.26%",
    "gpu_memory": "8GB",
}

# Old metrics to replace
OLD_METRICS = {
    "test_count": ["3,006", "3006", "2,898", "2898", "1,483", "1,448", "1483", "1448"],
    "optimization": ["6-13x", "6x-13x"],
    "validation_auc": ["93.98%", "94%"],
    "gpu_memory": ["12GB", "16GB"],
}

def find_markdown_files(root_dir: str = ".") -> List[Path]:
    """Find all markdown files in the repository."""
    root = Path(root_dir)
    exclude_dirs = {".git", "venv", "venv_gpu", "node_modules", ".hypothesis"}
    
    md_files = []
    for md_file in root.rglob("*.md"):
        # Skip if in excluded directory
        if any(excluded in md_file.parts for excluded in exclude_dirs):
            continue
        md_files.append(md_file)
    
    return md_files

def update_test_counts(content: str) -> Tuple[str, int]:
    """Update test count references."""
    changes = 0
    
    # Pattern: "1,483 tests" or "1,448 tests"
    for old_count in OLD_METRICS["test_count"]:
        pattern = rf'\b{re.escape(old_count)}\s+tests?\b'
        if re.search(pattern, content, re.IGNORECASE):
            content = re.sub(pattern, f"{CURRENT_METRICS['test_count']} tests", content, flags=re.IGNORECASE)
            changes += 1
    
    # Pattern: badge format "tests-1483%20total"
    for old_count in OLD_METRICS["test_count"]:
        old_no_comma = old_count.replace(",", "")
        pattern = rf'tests-{old_no_comma}%20total'
        if pattern in content:
            content = content.replace(pattern, f"tests-{CURRENT_METRICS['test_count_no_comma']}%20total")
            changes += 1
    
    return content, changes

def update_optimization_metrics(content: str) -> Tuple[str, int]:
    """Update optimization speedup references."""
    changes = 0
    
    for old_opt in OLD_METRICS["optimization"]:
        # Pattern: "6-13x" or "6x-13x"
        if old_opt in content:
            content = content.replace(old_opt, CURRENT_METRICS["optimization"])
            changes += 1
    
    return content, changes

def update_accuracy_metrics(content: str) -> Tuple[str, int]:
    """Update validation AUC references."""
    changes = 0
    
    for old_auc in OLD_METRICS["validation_auc"]:
        # Pattern: "93.98% test AUC" or "94% validation AUC"
        patterns = [
            (rf'\b{re.escape(old_auc)}\s+test\s+AUC\b', f"{CURRENT_METRICS['validation_auc']} validation AUC"),
            (rf'\b{re.escape(old_auc)}\s+validation\s+AUC\b', f"{CURRENT_METRICS['validation_auc']} validation AUC"),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                changes += 1
    
    return content, changes

def update_gpu_memory(content: str) -> Tuple[str, int]:
    """Update GPU memory requirements."""
    changes = 0
    
    for old_mem in OLD_METRICS["gpu_memory"]:
        # Pattern: "RTX 4070 (12GB)" -> "RTX 4070 (8GB)"
        pattern = rf'RTX\s+4070\s*\({re.escape(old_mem)}\)'
        if re.search(pattern, content):
            content = re.sub(pattern, f"RTX 4070 ({CURRENT_METRICS['gpu_memory']})", content)
            changes += 1
    
    return content, changes

def update_file(file_path: Path) -> int:
    """Update a single file with current metrics."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        total_changes = 0
        
        # Apply all updates
        content, changes = update_test_counts(content)
        total_changes += changes
        
        content, changes = update_optimization_metrics(content)
        total_changes += changes
        
        content, changes = update_accuracy_metrics(content)
        total_changes += changes
        
        content, changes = update_gpu_memory(content)
        total_changes += changes
        
        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ Updated {file_path} ({total_changes} changes)")
            return total_changes
        
        return 0
    
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return 0

def main():
    """Main update script."""
    print("🔍 Finding markdown files...")
    md_files = find_markdown_files()
    print(f"Found {len(md_files)} markdown files\n")
    
    print("📝 Updating metrics...")
    total_files_updated = 0
    total_changes = 0
    
    for md_file in md_files:
        changes = update_file(md_file)
        if changes > 0:
            total_files_updated += 1
            total_changes += changes
    
    print(f"\n✅ Complete!")
    print(f"📊 Updated {total_files_updated} files with {total_changes} total changes")
    print(f"\n📈 Current metrics:")
    print(f"  - Tests: {CURRENT_METRICS['test_count']}")
    print(f"  - Coverage: {CURRENT_METRICS['coverage']}")
    print(f"  - Optimization: {CURRENT_METRICS['optimization']}")
    print(f"  - Validation AUC: {CURRENT_METRICS['validation_auc']}")
    print(f"  - GPU Memory: {CURRENT_METRICS['gpu_memory']}")

if __name__ == "__main__":
    main()
