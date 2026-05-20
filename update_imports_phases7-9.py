"""Script to update imports for Phases 7-9 migration."""

import re
from pathlib import Path

REPLACEMENTS = [
    # Phase 7: Interpretability
    (r'from src\.interpretability\.', 'from src.features.interpretability.gradcam.'),
    (r'from src\.explainability\.', 'from src.features.interpretability.advanced.'),
    (r'from src\.visualization\.', 'from src.features.interpretability.visualization.'),
    
    # Phase 8: Research
    (r'from src\.annotation_interface\.', 'from src.features.research.annotation.'),
    (r'from src\.research_platform\.', 'from src.features.research.experiment.'),
    (r'from src\.hypothesis\.', 'from src.features.research.testing.'),
    
    # Phase 8: Advanced
    (r'from src\.causal\.', 'from src.features.advanced.causal.'),
    (r'from src\.discovery\.', 'from src.features.advanced.discovery.'),
    (r'from src\.omics\.', 'from src.features.advanced.omics.'),
    (r'from src\.spatial\.', 'from src.features.advanced.spatial.'),
    (r'from src\.cells\.', 'from src.features.advanced.cells.'),
    (r'from src\.multiscale\.', 'from src.features.advanced.multiscale.'),
    (r'from src\.segmentation\.', 'from src.features.advanced.segmentation.'),
    
    # Phase 9: Platform
    (r'from src\.monitoring\.', 'from src.platform.monitoring.'),
    (r'from src\.security\.', 'from src.platform.security.'),
    (r'from src\.database\.', 'from src.platform.database.'),
    (r'from src\.deployment\.', 'from src.platform.deployment.'),
    (r'from src\.cloud\.', 'from src.platform.cloud.'),
    (r'from src\.integration\.', 'from src.platform.integration.'),
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
        print(f"Error: {filepath}: {e}")
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
                print(f"✓ {filepath.relative_to(repo_root)}: {num_replacements}")
    
    print(f"\n{'='*60}")
    print(f"Phases 7-9 Import Update Complete")
    print(f"{'='*60}")
    print(f"Files scanned: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Replacements: {total_replacements}")

if __name__ == "__main__":
    main()
