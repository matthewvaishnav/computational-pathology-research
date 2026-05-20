"""Script to update imports for Phase 3 migration (models layer)."""

import re
from pathlib import Path

# Define import replacements
REPLACEMENTS = [
    # MIL models
    (r'from src\.models\.nnmil import', 'from src.models.mil.nnmil import'),
    (r'from src\.models\.attention_mil import', 'from src.models.mil.attention_mil import'),
    (r'from src\.models\.clam import', 'from src.models.mil.clam import'),
    (r'from src\.models\.transmil import', 'from src.models.mil.transmil import'),
    (r'from src\.models\.mil_base import', 'from src.models.mil.mil_base import'),
    (r'from src\.models\.instance_clustering import', 'from src.models.mil.instance_clustering import'),
    
    # TransnnMIL v2.0
    (r'from src\.models\.transnnmil_v2 import', 'from src.models.transnnmil.transnnmil_v2 import'),
    (r'from src\.models\.transnnmil import', 'from src.models.transnnmil.transnnmil import'),
    (r'from src\.models\.hierarchical_pooling import', 'from src.models.transnnmil.hierarchical_pooling import'),
    (r'from src\.models\.topology_branch import', 'from src.models.transnnmil.topology_branch import'),
    (r'from src\.models\.adaptive_pruning import', 'from src.models.transnnmil.adaptive_pruning import'),
    (r'from src\.models\.graph_cache import', 'from src.models.transnnmil.graph_cache import'),
    
    # Components
    (r'from src\.models\.attention_mechanisms import', 'from src.models.components.attention_mechanisms import'),
    (r'from src\.models\.encoders import', 'from src.models.components.encoders import'),
    (r'from src\.models\.heads import', 'from src.models.components.heads import'),
    (r'from src\.models\.feature_extractors import', 'from src.models.components.feature_extractors import'),
    (r'from src\.models\.fusion import', 'from src.models.components.fusion import'),
    (r'from src\.models\.fusion_strategies import', 'from src.models.components.fusion_strategies import'),
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
        repo_root / "models",  # Legacy models directory if exists
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
    print(f"Phase 3 Import Update Complete")
    print(f"{'='*60}")
    print(f"Total files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()
