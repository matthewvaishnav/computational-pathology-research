"""Script to update imports for Phase 2 migration (data layer)."""

import re
from pathlib import Path

# Define import replacements
REPLACEMENTS = [
    # Loaders
    (r'from src\.data\.loaders import', 'from src.data.loaders.loaders import'),
    (r'from src\.data\.bag_samplers import', 'from src.data.loaders.bag_samplers import'),
    (r'from src\.data\.batch_samplers import', 'from src.data.loaders.batch_samplers import'),
    (r'from src\.data\.prefetch import', 'from src.data.loaders.prefetch import'),
    
    # Datasets
    (r'from src\.data\.pcam_dataset import', 'from src.data.datasets.pcam_dataset import'),
    (r'from src\.data\.panda_dataset import', 'from src.data.datasets.panda_dataset import'),
    (r'from src\.data\.camelyon_dataset import', 'from src.data.datasets.camelyon_dataset import'),
    (r'from src\.data\.camelyon_annotations import', 'from src.data.datasets.camelyon_annotations import'),
    
    # WSI
    (r'from src\.data\.wsi_pipeline import', 'from src.data.wsi.pipeline import'),
    (r'from src\.data\.openslide_utils import', 'from src.data.wsi.openslide_utils import'),
    (r'from src\.data\.format_support import', 'from src.data.wsi.format_support import'),
    (r'from src\.streaming\.wsi_stream_reader import', 'from src.data.wsi.streaming import'),
    
    # Preprocessing
    (r'from src\.preprocessing\.', 'from src.data.preprocessing.'),
    (r'import src\.preprocessing\.', 'import src.data.preprocessing.'),
    (r'from src\.preprocessing import', 'from src.data.preprocessing import'),
    (r'import src\.preprocessing', 'import src.data.preprocessing'),
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
    print(f"Phase 2 Import Update Complete")
    print(f"{'='*60}")
    print(f"Total files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()
