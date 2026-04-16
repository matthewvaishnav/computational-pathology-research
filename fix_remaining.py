#!/usr/bin/env python3
"""Fix E731 lambda, F402 shadowing, E741 ambiguous names."""
from pathlib import Path
import re

# E731 lambda assignments from flake8 output
LAMBDA_FIXES = {
    'tests/dataset_testing/integration/test_pipeline_integration.py': [
        (211, 'get_train_size', 'lambda ds: len(ds.train_indices)'),
        (212, 'get_val_size', 'lambda ds: len(ds.val_indices)'),
    ],
}

# F402 import shadowing - rename loop variables
SHADOWING_FIXES = {
    'tests/dataset_testing/property_based/test_openslide_properties.py': [
        (106, 'patch', 'mock_patch'),
        (230, 'patch', 'mock_patch'),
    ],
    'tests/dataset_testing/unit/test_openslide_integration.py': [
        (160, 'patch', 'mock_patch'),
        (205, 'patch', 'mock_patch'),
        (464, 'patch', 'mock_patch'),
    ],
    'tests/dataset_testing/unit/test_openslide_tissue_detection.py': [
        (149, 'patch', 'mock_patch'),
        (214, 'patch', 'mock_patch'),
        (275, 'patch', 'mock_patch'),
    ],
    'tests/dataset_testing/unit/test_openslide_utils.py': [
        (244, 'patch', 'mock_patch'),
        (550, 'patch', 'mock_patch'),
        (584, 'patch', 'mock_patch'),
    ],
}

# E741 ambiguous variable names
AMBIGUOUS_FIXES = {
    'tests/dataset_testing/property_based/test_openslide_properties.py': [
        (196, 'l', 'level'),
    ],
    'tests/dataset_testing/synthetic/multimodal_generator.py': [
        (491, 'l', 'label'),
    ],
}

def fix_lambda_assignments(filepath, fixes):
    """Convert lambda assignments to def functions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for lineno, func_name, lambda_expr in fixes:
        idx = lineno - 1
        if idx < len(lines):
            line = lines[idx]
            # Extract lambda body
            match = re.search(r'lambda\s+([^:]+):\s*(.+)', lambda_expr)
            if match:
                params, body = match.groups()
                # Replace with def function
                indent = len(line) - len(line.lstrip())
                new_lines = [
                    ' ' * indent + f'def {func_name}({params}):\n',
                    ' ' * (indent + 4) + f'return {body}\n',
                ]
                lines[idx] = ''.join(new_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def fix_shadowing(filepath, fixes):
    """Rename loop variables that shadow imports."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for lineno, old_name, new_name in fixes:
        # Find the line and replace
        lines = content.split('\n')
        if lineno - 1 < len(lines):
            line = lines[lineno - 1]
            # Replace in for loop
            lines[lineno - 1] = line.replace(f'for {old_name} in', f'for {new_name} in')
            # Also replace usage in the loop body (next few lines)
            for i in range(lineno, min(lineno + 10, len(lines))):
                if lines[i].strip() and not lines[i].strip().startswith('for'):
                    lines[i] = re.sub(rf'\b{old_name}\b', new_name, lines[i])
                if lines[i].strip().startswith(('def ', 'class ', 'for ', '@')):
                    break
        content = '\n'.join(lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def fix_ambiguous(filepath, fixes):
    """Rename ambiguous variable names."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for lineno, old_name, new_name in fixes:
        idx = lineno - 1
        if idx < len(lines):
            # Replace in the specific line and context
            for i in range(max(0, idx - 2), min(len(lines), idx + 10)):
                lines[i] = re.sub(rf'\b{old_name}\b', new_name, lines[i])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    """Apply all fixes."""
    fixed = 0
    
    # Fix lambda assignments
    for filepath, fixes in LAMBDA_FIXES.items():
        path = Path(filepath)
        if path.exists():
            fix_lambda_assignments(path, fixes)
            fixed += 1
            print(f"Fixed E731 lambdas: {filepath}")
    
    # Fix shadowing
    for filepath, fixes in SHADOWING_FIXES.items():
        path = Path(filepath)
        if path.exists():
            fix_shadowing(path, fixes)
            fixed += 1
            print(f"Fixed F402 shadowing: {filepath}")
    
    # Fix ambiguous names
    for filepath, fixes in AMBIGUOUS_FIXES.items():
        path = Path(filepath)
        if path.exists():
            fix_ambiguous(path, fixes)
            fixed += 1
            print(f"Fixed E741 ambiguous: {filepath}")
    
    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
