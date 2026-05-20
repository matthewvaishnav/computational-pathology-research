"""Script to update imports for Phase 6 migration (clinical features)."""

import re
from pathlib import Path

REPLACEMENTS = [
    # Clinical workflow
    (r'from src\.clinical\.', 'from src.features.clinical.workflow.'),
    (r'import src\.clinical\.', 'import src.features.clinical.workflow.'),
    (r'from src\.clinical import', 'from src.features.clinical.workflow import'),
    (r'import src\.clinical', 'import src.features.clinical.workflow'),
    
    # PACS
    (r'from src\.pacs\.', 'from src.features.clinical.pacs.'),
    (r'import src\.pacs\.', 'import src.features.clinical.pacs.'),
    (r'from src\.pacs import', 'from src.features.clinical.pacs import'),
    (r'import src\.pacs', 'import src.features.clinical.pacs'),
    
    # Clinical validation
    (r'from src\.clinical_validation\.', 'from src.features.clinical.validation.'),
    (r'import src\.clinical_validation\.', 'import src.features.clinical.validation.'),
    (r'from src\.clinical_validation import', 'from src.features.clinical.validation import'),
    (r'import src\.clinical_validation', 'import src.features.clinical.validation'),
]

def update_file(filepath: Path) -> tuple[int, bool]:
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        num_replacements = 0
        
        for pattern, replacement in REPLACEMENTS:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                num_replacements += count
        
        if content != original_content:
            filepath.write_text(content, encoding='utf-8')
            return num_replacements, True
        
        return 0, False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, False

def main():
    repo_root = Path(__file__).parent
    
    dirs_to_process = [
        repo_root / "src",
        repo_root / "tests",
        repo_root / "scripts",
        repo_root / "experiments",
    ]
    
    total_files = 0
    modified_files = 0
    total_replacements = 0
    
    for directory in dirs_to_process:
        if not directory.exists():
            continue
        
        for filepath in directory.rglob("*.py"):
            if "__pycache__" in str(filepath) or ".egg-info" in str(filepath):
                continue
            
            total_files += 1
            num_replacements, modified = update_file(filepath)
            
            if modified:
                modified_files += 1
                total_replacements += num_replacements
                print(f"✓ {filepath.relative_to(repo_root)}: {num_replacements} replacements")
    
    print(f"\n{'='*60}")
    print(f"Phase 6 Import Update Complete")
    print(f"{'='*60}")
    print(f"Total files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()
