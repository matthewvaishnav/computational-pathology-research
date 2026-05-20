"""Complete remaining phases 7-10 of architecture migration."""

import shutil
from pathlib import Path

def main():
    repo_root = Path(__file__).parent
    src = repo_root / "src"
    
    print("="*60)
    print("Phase 7: Interpretability Features")
    print("="*60)
    
    # Create features/interpretability
    (src / "features" / "interpretability").mkdir(exist_ok=True)
    
    # Move directories
    moves_p7 = [
        ("interpretability", "features/interpretability/gradcam"),
        ("explainability", "features/interpretability/advanced"),
        ("visualization", "features/interpretability/visualization"),
    ]
    
    for old, new in moves_p7:
        old_path = src / old
        new_path = src / new
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved {old} → {new}")
    
    # Create __init__.py
    (src / "features" / "interpretability" / "__init__.py").write_text(
        '"""Interpretability and explainability features."""\n\n__all__ = ["gradcam", "advanced", "visualization"]\n'
    )
    
    print("\n" + "="*60)
    print("Phase 8: Research & Advanced Features")
    print("="*60)
    
    # Create features/research and features/advanced
    (src / "features" / "research").mkdir(exist_ok=True)
    (src / "features" / "advanced").mkdir(exist_ok=True)
    
    # Move research directories
    moves_p8_research = [
        ("annotation_interface", "features/research/annotation"),
        ("research_platform", "features/research/experiment"),
        ("hypothesis", "features/research/testing"),
    ]
    
    for old, new in moves_p8_research:
        old_path = src / old
        new_path = src / new
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved {old} → {new}")
    
    # Move advanced directories
    moves_p8_advanced = [
        ("causal", "features/advanced/causal"),
        ("discovery", "features/advanced/discovery"),
        ("omics", "features/advanced/omics"),
        ("spatial", "features/advanced/spatial"),
        ("cells", "features/advanced/cells"),
        ("multiscale", "features/advanced/multiscale"),
        ("segmentation", "features/advanced/segmentation"),
    ]
    
    for old, new in moves_p8_advanced:
        old_path = src / old
        new_path = src / new
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved {old} → {new}")
    
    # Create __init__.py files
    (src / "features" / "research" / "__init__.py").write_text(
        '"""Research platform features."""\n\n__all__ = ["annotation", "experiment", "testing"]\n'
    )
    (src / "features" / "advanced" / "__init__.py").write_text(
        '"""Advanced analysis features."""\n\n__all__ = ["causal", "discovery", "omics", "spatial", "cells", "multiscale", "segmentation"]\n'
    )
    
    print("\n" + "="*60)
    print("Phase 9: Platform Services")
    print("="*60)
    
    # Create platform directory
    (src / "platform").mkdir(exist_ok=True)
    
    # Move platform directories
    moves_p9 = [
        ("monitoring", "platform/monitoring"),
        ("security", "platform/security"),
        ("database", "platform/database"),
        ("deployment", "platform/deployment"),
        ("cloud", "platform/cloud"),
        ("integration", "platform/integration"),
    ]
    
    for old, new in moves_p9:
        old_path = src / old
        new_path = src / new
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved {old} → {new}")
    
    # Create __init__.py
    (src / "platform" / "__init__.py").write_text(
        '"""Platform services."""\n\n__all__ = ["monitoring", "security", "database", "deployment", "cloud", "integration"]\n'
    )
    
    print("\n" + "="*60)
    print("Phase 10: Cleanup")
    print("="*60)
    
    # Update features __init__.py
    (src / "features" / "__init__.py").write_text(
        '"""Domain-specific features for computational pathology."""\n\n__all__ = ["federated", "clinical", "interpretability", "research", "advanced"]\n'
    )
    
    print("✓ Updated features/__init__.py")
    print("✓ Migration structure complete!")
    
    print("\n" + "="*60)
    print("MIGRATION COMPLETE")
    print("="*60)
    print("Phases 7-10 executed successfully")
    print("Next: Run import updaters for phases 7-9")

if __name__ == "__main__":
    main()
