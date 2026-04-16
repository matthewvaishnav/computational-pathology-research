#!/usr/bin/env python3
"""Quick fix for common flake8 violations without running flake8."""
import re
from pathlib import Path

def fix_file(filepath):
    """Fix common violations in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Fix E203: whitespace before ':' in slicing
        content = re.sub(r'(\w+)\s+:', r'\1:', content)
        
        # Fix W293: blank line contains whitespace
        content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Fix W291: trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Fix E741: ambiguous variable name 'l'
        # This needs manual review, skip for now
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all Python files."""
    fixed = 0
    for pattern in ['src/**/*.py', 'tests/**/*.py', 'experiments/**/*.py']:
        for filepath in Path('.').glob(pattern):
            if fix_file(filepath):
                fixed += 1
                print(f"Fixed: {filepath}")
    
    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
