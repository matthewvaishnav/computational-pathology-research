"""Fix relative imports in Phase 3 moved files."""

import re
from pathlib import Path

def fix_file(filepath: Path) -> tuple[int, bool]:
    """Fix relative imports in a file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        num_replacements = 0
        
        # Fix relative imports based on file location
        if "src/models/mil/" in str(filepath):
            # MIL files: fix relative imports to components and mil
            replacements = [
                (r'from \.attention_mechanisms import', 'from src.models.components.attention_mechanisms import'),
                (r'from \.fusion_strategies import', 'from src.models.components.fusion_strategies import'),
                (r'from \.encoders import', 'from src.models.components.encoders import'),
                (r'from \.heads import', 'from src.models.components.heads import'),
                (r'from \.mil_base import', 'from src.models.mil.mil_base import'),
                (r'from \.instance_clustering import', 'from src.models.mil.instance_clustering import'),
            ]
        elif "src/models/transnnmil/" in str(filepath):
            # TransnnMIL files: fix relative imports
            replacements = [
                (r'from \.attention_mechanisms import', 'from src.models.components.attention_mechanisms import'),
                (r'from \.encoders import', 'from src.models.components.encoders import'),
                (r'from \.heads import', 'from src.models.components.heads import'),
                (r'from \.hierarchical_pooling import', 'from src.models.transnnmil.hierarchical_pooling import'),
                (r'from \.topology_branch import', 'from src.models.transnnmil.topology_branch import'),
                (r'from \.adaptive_pruning import', 'from src.models.transnnmil.adaptive_pruning import'),
                (r'from \.graph_cache import', 'from src.models.transnnmil.graph_cache import'),
            ]
        elif "src/models/components/" in str(filepath):
            # Components files: fix relative imports
            replacements = [
                (r'from \.attention_mechanisms import', 'from src.models.components.attention_mechanisms import'),
                (r'from \.encoders import', 'from src.models.components.encoders import'),
                (r'from \.heads import', 'from src.models.components.heads import'),
                (r'from \.fusion import', 'from src.models.components.fusion import'),
            ]
        else:
            return 0, False
        
        for pattern, replacement in replacements:
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
    """Fix relative imports in moved model files."""
    repo_root = Path(__file__).parent
    
    dirs_to_process = [
        repo_root / "src" / "models" / "mil",
        repo_root / "src" / "models" / "transnnmil",
        repo_root / "src" / "models" / "components",
    ]
    
    total_files = 0
    modified_files = 0
    total_replacements = 0
    
    for directory in dirs_to_process:
        if not directory.exists():
            continue
        
        for filepath in directory.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue
            
            total_files += 1
            num_replacements, modified = fix_file(filepath)
            
            if modified:
                modified_files += 1
                total_replacements += num_replacements
                print(f"✓ {filepath.relative_to(repo_root)}: {num_replacements} replacements")
    
    print(f"\n{'='*60}")
    print(f"Relative Import Fix Complete")
    print(f"{'='*60}")
    print(f"Total files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Total replacements: {total_replacements}")

if __name__ == "__main__":
    main()
