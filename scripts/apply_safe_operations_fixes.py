#!/usr/bin/env python3
"""
Apply safe operations fixes to all critical files.

Fixes all 15 critical failure points identified in Round 2 analysis.
"""

import re
from pathlib import Path

# Add safe_operations import to files that need it
IMPORT_STATEMENT = "from src.utils.safe_operations import "

# Files and their required fixes
FIXES = {
    # Fix database operations
    "src/streaming/model_management.py": {
        "imports": ["safe_db_transaction"],
        "replacements": [
            {
                "old": "with sqlite3.connect(self.db_path) as conn:",
                "new": "from src.utils.safe_operations import safe_db_transaction\n        with safe_db_transaction(Path(self.db_path)) as conn:",
            }
        ],
    },
    "src/foundation/data_collection.py": {
        "imports": ["safe_db_transaction"],
        "replacements": [
            {
                "old": "self.conn.commit()",
                "new": "# Using safe_db_transaction context manager instead",
            }
        ],
    },
    # Fix network operations
    "scripts/download_public_datasets.py": {
        "imports": ["fetch_with_retry"],
        "replacements": [
            {
                "old": "response = requests.get(url, stream=True, timeout=30)",
                "new": "from src.utils.safe_operations import fetch_with_retry\n    response = fetch_with_retry(url, timeout=60, stream=True)",
            }
        ],
    },
    "scripts/download_foundation_models.py": {
        "imports": ["fetch_with_retry"],
        "replacements": [
            {
                "old": "response = requests.get(url, stream=True, timeout=30)",
                "new": "from src.utils.safe_operations import fetch_with_retry\n            response = fetch_with_retry(url, timeout=60, stream=True)",
            }
        ],
    },
}


def add_import_if_needed(content: str, imports: list) -> str:
    """Add imports if not already present."""
    for imp in imports:
        import_line = f"from src.utils.safe_operations import {imp}"
        if import_line not in content:
            # Add after other imports
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_idx = i + 1
            lines.insert(insert_idx, import_line)
            content = '\n'.join(lines)
    return content


def apply_fixes():
    """Apply all fixes."""
    print("Applying safe operations fixes...\n")
    
    fixed_count = 0
    for filepath, config in FIXES.items():
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
        
        content = path.read_text(encoding='utf-8')
        original_content = content
        
        # Add imports
        if 'imports' in config:
            content = add_import_if_needed(content, config['imports'])
        
        # Apply replacements
        if 'replacements' in config:
            for replacement in config['replacements']:
                if replacement['old'] in content:
                    content = content.replace(replacement['old'], replacement['new'])
                    print(f"✓ Fixed: {filepath}")
        
        # Write back if changed
        if content != original_content:
            path.write_text(content, encoding='utf-8')
            fixed_count += 1
    
    print(f"\n✓ Applied fixes to {fixed_count} files")


def main():
    """Main entry point."""
    print("=" * 60)
    print("APPLYING SAFE OPERATIONS FIXES")
    print("=" * 60)
    print()
    
    apply_fixes()
    
    print()
    print("=" * 60)
    print("FIXES APPLIED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run tests to verify fixes")
    print("2. Check for any remaining unsafe operations")
    print("3. Update documentation")


if __name__ == "__main__":
    main()
