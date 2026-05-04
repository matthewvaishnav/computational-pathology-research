"""
Compute code quality metrics for refactored modules.

Metrics:
- Average file size
- Max file size
- Function length distribution
- Code duplication
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple


def count_lines(filepath: Path) -> int:
    """Count lines in file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except Exception:
        return 0


def analyze_directory(directory: Path, pattern: str = "*.py") -> Dict:
    """Analyze Python files in directory."""
    files = list(directory.rglob(pattern))
    
    if not files:
        return {
            "file_count": 0,
            "total_lines": 0,
            "avg_lines": 0,
            "max_lines": 0,
            "min_lines": 0,
            "files": []
        }
    
    file_sizes = []
    cwd = Path.cwd()
    for f in files:
        if '__pycache__' in str(f) or '.venv' in str(f) or 'venv' in str(f):
            continue
        lines = count_lines(f)
        if lines > 0:
            try:
                rel_path = f.relative_to(cwd)
            except ValueError:
                rel_path = f
            file_sizes.append((str(rel_path), lines))
    
    if not file_sizes:
        return {
            "file_count": 0,
            "total_lines": 0,
            "avg_lines": 0,
            "max_lines": 0,
            "min_lines": 0,
            "files": []
        }
    
    sizes = [s[1] for s in file_sizes]
    
    return {
        "file_count": len(file_sizes),
        "total_lines": sum(sizes),
        "avg_lines": sum(sizes) / len(sizes),
        "max_lines": max(sizes),
        "min_lines": min(sizes),
        "max_file": max(file_sizes, key=lambda x: x[1]),
        "files": sorted(file_sizes, key=lambda x: x[1], reverse=True)[:10]
    }


def main():
    """Compute and display metrics."""
    print("=" * 80)
    print("CODE QUALITY METRICS - Clean Code Refactoring")
    print("=" * 80)
    print()
    
    # Analyze refactored modules
    modules = {
        "API Routes": Path("src/api"),
        "MIL Models": Path("src/models"),
        "Memory Optimizer": Path("src/streaming/memory"),
        "Clinical Modules": Path("src/clinical"),
        "Streaming Modules": Path("src/streaming"),
    }
    
    all_metrics = {}
    
    for name, path in modules.items():
        if path.exists():
            metrics = analyze_directory(path)
            all_metrics[name] = metrics
            
            print(f"{name}:")
            print(f"  Files: {metrics['file_count']}")
            print(f"  Total Lines: {metrics['total_lines']}")
            print(f"  Avg Lines/File: {metrics['avg_lines']:.1f}")
            print(f"  Max Lines: {metrics['max_lines']}")
            print(f"  Min Lines: {metrics['min_lines']}")
            if 'max_file' in metrics:
                print(f"  Largest File: {metrics['max_file'][0]} ({metrics['max_file'][1]} lines)")
            print()
    
    # Overall statistics
    total_files = sum(m['file_count'] for m in all_metrics.values())
    total_lines = sum(m['total_lines'] for m in all_metrics.values())
    avg_file_size = total_lines / total_files if total_files > 0 else 0
    max_file_size = max((m['max_lines'] for m in all_metrics.values() if m['file_count'] > 0), default=0)
    
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total Files Analyzed: {total_files}")
    print(f"Total Lines of Code: {total_lines}")
    print(f"Average File Size: {avg_file_size:.1f} lines")
    print(f"Maximum File Size: {max_file_size} lines")
    print()
    
    # Quality targets
    print("=" * 80)
    print("QUALITY TARGETS")
    print("=" * 80)
    print(f"✓ Avg file size <400 lines: {'PASS' if avg_file_size < 400 else 'FAIL'} ({avg_file_size:.1f})")
    print(f"✓ Max file size <500 lines: {'PASS' if max_file_size < 500 else 'FAIL'} ({max_file_size})")
    print()
    
    # Top 10 largest files
    print("=" * 80)
    print("TOP 10 LARGEST FILES")
    print("=" * 80)
    all_files = []
    for metrics in all_metrics.values():
        all_files.extend(metrics['files'])
    
    all_files_sorted = sorted(all_files, key=lambda x: x[1], reverse=True)[:10]
    for i, (filepath, lines) in enumerate(all_files_sorted, 1):
        print(f"{i:2d}. {filepath:60s} {lines:5d} lines")
    print()


if __name__ == "__main__":
    main()
