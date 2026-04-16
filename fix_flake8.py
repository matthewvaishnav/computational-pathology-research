#!/usr/bin/env python3
"""Fix remaining flake8 violations."""
import subprocess
import re
from pathlib import Path

def remove_unused_imports():
    """Remove F401 unused imports using autoflake."""
    print("Removing unused imports (F401)...")
    subprocess.run([
        "python", "-m", "autoflake",
        "--in-place",
        "--remove-all-unused-imports",
        "--recursive",
        "src/", "tests/", "experiments/"
    ])

def fix_lambda_assignments():
    """Fix E731 lambda assignments."""
    print("Fixing E731 lambda assignments...")
    # Get list of files with E731 violations
    result = subprocess.run(
        ["python", "-m", "flake8", "--select=E731", "src/", "tests/", "experiments/"],
        capture_output=True, text=True
    )
    
    for line in result.stdout.splitlines():
        if "E731" in line:
            print(f"  Found: {line}")

def fix_import_shadowing():
    """Fix F402 import shadowing."""
    print("Fixing F402 import shadowing...")
    # Get list of files with F402 violations
    result = subprocess.run(
        ["python", "-m", "flake8", "--select=F402", "tests/"],
        capture_output=True, text=True
    )
    
    files_to_fix = {}
    for line in result.stdout.splitlines():
        if "F402" in line:
            match = re.match(r"(.+):(\d+):\d+: F402 import '(\w+)' from line \d+ shadowed by loop variable", line)
            if match:
                filepath, lineno, varname = match.groups()
                if filepath not in files_to_fix:
                    files_to_fix[filepath] = []
                files_to_fix[filepath].append((int(lineno), varname))
    
    for filepath, violations in files_to_fix.items():
        print(f"  Fixing {filepath}")
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for lineno, varname in violations:
            line = lines[lineno - 1]
            # Rename loop variable
            new_varname = f"{varname}_item"
            lines[lineno - 1] = line.replace(f"for {varname} in", f"for {new_varname} in")
        
        with open(filepath, 'w') as f:
            f.writelines(lines)

if __name__ == "__main__":
    # Try autoflake for unused imports
    try:
        remove_unused_imports()
    except Exception as e:
        print(f"autoflake not available: {e}")
    
    fix_lambda_assignments()
    fix_import_shadowing()
    
    print("\nDone! Run flake8 to check remaining violations.")
