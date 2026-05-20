"""Script to update imports for Phase 5 migration (federated features)."""

import re
from pathlib import Path

# Define import replacements
REPLACEMENTS = [
    # Federated → PathologyFL
    (r'from src\.federated\.', 'from src.features.federated.pathology_fl.'),
    (r'import src\.federated\.', 'import src.features.federated.pathology_fl.'),
    (r'from src\.federated import', 'from src.features.federated.pathology_fl import'),
    (r'import src\.federated', 'import src.features.federated.pathology_fl'),
    
    # DMI
    (r'from src\.dmi\.', 'from src.features.federated.dmi.'),
    (r'import src\.dmi\.', 'import src.features.federated.dmi.'),
    (r'from src\.dmi import', 'from src.features.federated.dmi import'),
    (r'import src\.dmi', 'import src.features.federated.dmi'),
    
    # CPI
    (r'from src\.cpi\.', 'from src.features.federated.cpi.'),
    (r'import src\.cpi\.', 'import src.features.federated.cpi.'),
    (r'from src\.cpi import', 'from src.features.federated.cpi import'),
    (r'import src\.cpi', 'import src.features.federated.cpi'),
    
    # IMR
    (r'from src\.imr\.', 'from src.features.federated.imr.'),
    (r'import src\.imr\.', 'import src.features.federated.imr.'),
    (r'from src\.imr import', 'from src.features.federated.imr import'),
    (r'import src\.imr', 'import src.features.federated.imr'),
    
    # MKN
    (r'from src\.mkn\.', 'from src.features.federated.mkn.'),
    (r'import src\.mkn\.', 'import src.features.federated.mkn.'),
    (r'from src\.mkn import', 'from src.features.federated.mkn import'),
    (r'import src\.mkn', 'import src.features.federated.mkn'),
]

def update_file(filepath: Path) -> tuple[int, bool]:
    """Update imports in a single file.
    
    Returns:
        (num_replacements, file_modified)
    """
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
    """Update all Python files in the repository."""
    repo_root = Path(__file__).parent
    
    # Directories to process
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
            # Skip __pycache__ and .egg-info
            if "__pycache__" in str(filepath) or ".egg-info" in str(filepath):
                continue
            
            total_files += 1
            num_replacements, modified = update_file(filepath)
            
            if modified:
                modified_files += 1
                total_replacements += num_replacements
                print(f"✓ {filepath.relative_to(repo_root)}: {num_replacements} replacements")
    
    print(f"\n{'='*60}")
    print(f"Phase 5 Import Update Complete")
    print(f"{'='*60}")
    print(f"Total files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()
