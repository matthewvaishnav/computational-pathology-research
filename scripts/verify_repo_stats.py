#!/usr/bin/env python3
"""
Verify repository statistics for accurate reporting.

Counts:
- Lines of code (Python only)
- Number of Python modules
- Number of test files
- Total files by type
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import json


def count_lines_in_file(filepath):
    """Count non-empty, non-comment lines in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
        
        return {
            'total': len(lines),
            'code': code_lines,
            'comments': comment_lines,
            'blank': blank_lines
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return {'total': 0, 'code': 0, 'comments': 0, 'blank': 0}


def analyze_repository(root_dir):
    """Analyze repository structure and count statistics."""
    root_path = Path(root_dir)
    
    # Directories to exclude
    exclude_dirs = {
        '.git', '.hypothesis', '__pycache__', '.pytest_cache',
        'node_modules', '.venv', 'venv', 'env',
        '.kilo', '.kiro', 'build', 'dist', '*.egg-info'
    }
    
    stats = {
        'python_files': 0,
        'python_modules': 0,
        'test_files': 0,
        'total_lines': 0,
        'code_lines': 0,
        'comment_lines': 0,
        'blank_lines': 0,
        'files_by_extension': defaultdict(int),
        'directories': set(),
        'python_files_list': [],
        'test_files_list': []
    }
    
    for filepath in root_path.rglob('*'):
        # Skip excluded directories
        if any(excluded in filepath.parts for excluded in exclude_dirs):
            continue
        
        if filepath.is_file():
            extension = filepath.suffix
            stats['files_by_extension'][extension] += 1
            
            # Count Python files
            if extension == '.py':
                stats['python_files'] += 1
                stats['python_files_list'].append(str(filepath.relative_to(root_path)))
                
                # Count lines
                line_counts = count_lines_in_file(filepath)
                stats['total_lines'] += line_counts['total']
                stats['code_lines'] += line_counts['code']
                stats['comment_lines'] += line_counts['comments']
                stats['blank_lines'] += line_counts['blank']
                
                # Check if it's a module (not a script)
                if '__init__.py' in filepath.name or filepath.parent.name in {'src', 'models', 'training', 'data'}:
                    stats['python_modules'] += 1
                
                # Check if it's a test file
                if 'test' in filepath.name or 'tests' in filepath.parts:
                    stats['test_files'] += 1
                    stats['test_files_list'].append(str(filepath.relative_to(root_path)))
        
        elif filepath.is_dir():
            stats['directories'].add(str(filepath.relative_to(root_path)))
    
    stats['directories'] = len(stats['directories'])
    
    return stats


def format_number(num):
    """Format number with thousands separator."""
    return f"{num:,}"


def print_report(stats):
    """Print formatted statistics report."""
    print("=" * 70)
    print("REPOSITORY STATISTICS VERIFICATION")
    print("=" * 70)
    print()
    
    print("📊 CODE METRICS")
    print("-" * 70)
    print(f"Python files:           {format_number(stats['python_files'])}")
    print(f"Python modules:         {format_number(stats['python_modules'])}")
    print(f"Test files:             {format_number(stats['test_files'])}")
    print(f"Directories:            {format_number(stats['directories'])}")
    print()
    
    print("📝 LINE COUNTS (Python only)")
    print("-" * 70)
    print(f"Total lines:            {format_number(stats['total_lines'])}")
    print(f"Code lines:             {format_number(stats['code_lines'])}")
    print(f"Comment lines:          {format_number(stats['comment_lines'])}")
    print(f"Blank lines:            {format_number(stats['blank_lines'])}")
    print()
    
    print("📁 FILES BY EXTENSION (Top 10)")
    print("-" * 70)
    sorted_extensions = sorted(
        stats['files_by_extension'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for ext, count in sorted_extensions[:10]:
        ext_display = ext if ext else '(no extension)'
        print(f"{ext_display:20s} {format_number(count):>10s}")
    print()
    
    print("✅ VERIFICATION SUMMARY")
    print("-" * 70)
    print(f"Claimed LOC:            ~195,000")
    print(f"Actual code lines:      {format_number(stats['code_lines'])}")
    print(f"Match:                  {'✅ Yes' if abs(stats['code_lines'] - 195000) < 50000 else '❌ No'}")
    print()
    print(f"Claimed test files:     5,071+")
    print(f"Actual test files:      {format_number(stats['test_files'])}")
    print(f"Match:                  {'✅ Yes' if stats['test_files'] >= 5000 else '❌ No'}")
    print()
    print(f"Claimed modules:        544")
    print(f"Actual modules:         {format_number(stats['python_modules'])}")
    print(f"Match:                  {'✅ Yes' if abs(stats['python_modules'] - 544) < 100 else '❌ No'}")
    print()
    
    print("=" * 70)


def save_detailed_report(stats, output_file):
    """Save detailed statistics to JSON file."""
    # Convert sets to lists for JSON serialization
    output_stats = {
        k: list(v) if isinstance(v, set) else v
        for k, v in stats.items()
    }
    
    # Convert defaultdict to regular dict
    output_stats['files_by_extension'] = dict(output_stats['files_by_extension'])
    
    with open(output_file, 'w') as f:
        json.dump(output_stats, f, indent=2)
    
    print(f"Detailed report saved to: {output_file}")


def main():
    """Main entry point."""
    # Get repository root (parent of scripts directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    print(f"Analyzing repository: {repo_root}")
    print()
    
    # Analyze repository
    stats = analyze_repository(repo_root)
    
    # Print report
    print_report(stats)
    
    # Save detailed report
    output_file = repo_root / 'docs' / 'VERIFIED_METRICS.json'
    save_detailed_report(stats, output_file)
    
    # Create markdown report
    md_file = repo_root / 'docs' / 'VERIFIED_METRICS.md'
    with open(md_file, 'w') as f:
        f.write("# Verified Repository Metrics\n\n")
        f.write("**Generated:** " + str(Path(__file__).stat().st_mtime) + "\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Python files:** {format_number(stats['python_files'])}\n")
        f.write(f"- **Python modules:** {format_number(stats['python_modules'])}\n")
        f.write(f"- **Test files:** {format_number(stats['test_files'])}\n")
        f.write(f"- **Code lines:** {format_number(stats['code_lines'])}\n")
        f.write(f"- **Total lines:** {format_number(stats['total_lines'])}\n\n")
        f.write("## Methodology\n\n")
        f.write("- **Python files:** All `.py` files excluding `.git`, `.hypothesis`, `__pycache__`, etc.\n")
        f.write("- **Code lines:** Non-blank, non-comment lines in Python files\n")
        f.write("- **Test files:** Files with 'test' in name or in 'tests' directory\n")
        f.write("- **Modules:** Python files in `src/`, `models/`, `training/`, `data/` or `__init__.py` files\n\n")
        f.write("See `VERIFIED_METRICS.json` for detailed breakdown.\n")
    
    print(f"Markdown report saved to: {md_file}")


if __name__ == '__main__':
    main()
