#!/usr/bin/env python3
"""Remove unused imports by reading files directly."""
import re
from pathlib import Path

# Known unused imports from flake8 output
UNUSED_IMPORTS = {
    'tests/dataset_testing/base_interfaces.py': ['torch'],
    'tests/dataset_testing/hypothesis_strategies.py': ['tempfile'],
    'tests/dataset_testing/integration/test_pipeline_integration.py': ['tempfile', 'shutil'],
    'tests/dataset_testing/integration/test_training_loop_integration.py': ['pytest', 'Path', 'h5py', 'Dict', 'Any'],
    'tests/dataset_testing/pcam/test_pcam_download.py': ['pytest', 'Mock', 'patch', 'MagicMock', 'Optional', 'requests', 'urllib.error'],
    'tests/dataset_testing/pcam/test_pcam_properties.py': ['pytest', 'torch', 'Dict', 'Any', 'arrays', 'configuration_strategy'],
    'tests/dataset_testing/pcam/test_pcam_unit.py': ['patch', 'MagicMock', 'torchvision.transforms'],
    'tests/dataset_testing/performance/test_caching_optimization.py': ['torch', 'Path', 'tempfile', 'shutil', 'Dict', 'Any'],
    'tests/dataset_testing/performance/test_performance_benchmarks.py': ['Path', 'Dict', 'Any', 'tempfile', 'shutil'],
    'tests/dataset_testing/property_based/test_camelyon_properties.py': ['Dict', 'Any'],
    'tests/dataset_testing/property_based/test_multimodal_properties.py': ['Optional', 'numpy as np'],
    'tests/dataset_testing/property_based/test_openslide_properties.py': ['pytest', 'wsi_coordinates_strategy', 'pyramid_level_strategy'],
    'tests/dataset_testing/synthetic/camelyon_generator.py': ['torch', 'Optional', 'List'],
    'tests/dataset_testing/synthetic/multimodal_generator.py': ['torch', 'Optional'],
    'tests/dataset_testing/synthetic/pcam_generator.py': ['torch', 'Optional'],
    'tests/dataset_testing/synthetic/wsi_generator.py': ['Optional'],
    'tests/dataset_testing/unit/test_batch_preprocessing.py': ['Mock', 'patch', 'h5py'],
    'tests/dataset_testing/unit/test_camelyon_error_handling.py': ['json', 'tempfile', 'Path', 'Dict', 'Any', 'torch'],
    'tests/dataset_testing/unit/test_camelyon_patient_splits.py': ['json', 'tempfile', 'Path', 'Dict', 'Any', 'List', 'h5py', 'numpy as np', 'torch'],
    'tests/dataset_testing/unit/test_camelyon_unit.py': ['json', 'tempfile', 'Path', 'Dict', 'Any'],
    'tests/dataset_testing/unit/test_error_handling.py': ['Mock', 'patch', 'MagicMock', 'io'],
    'tests/dataset_testing/unit/test_multimodal_dataset.py': ['json', 'Dict', 'Any', 'Optional', 'patch', 'MagicMock', 'h5py', 'numpy as np', 'pytest'],
    'tests/dataset_testing/unit/test_multimodal_missing_data.py': ['json', 'Dict', 'Any', 'List', 'Optional', 'patch', 'MagicMock', 'h5py', 'numpy as np', 'pytest'],
    'tests/dataset_testing/unit/test_network_storage_constraints.py': ['Mock', 'h5py', 'batch_save_to_hdf5'],
    'tests/dataset_testing/unit/test_openslide_error_handling.py': ['MagicMock', 'Optional', 'Tuple', 'os', 'get_slide_info', 'WSISyntheticSpec'],
    'tests/dataset_testing/unit/test_openslide_integration.py': ['MagicMock', 'get_slide_info', 'check_openslide_available', 'WSISyntheticGenerator', 'WSISyntheticSpec', 'ErrorSimulator'],
    'tests/dataset_testing/unit/test_openslide_properties.py': ['pytest', 'numpy as np', 'MagicMock', 'Optional', 'Tuple', 'List', 'PIL.Image', 'check_openslide_available', 'WSISyntheticSpec'],
    'tests/dataset_testing/unit/test_openslide_tissue_detection.py': ['pytest', 'MagicMock', 'Dict', 'Any', 'Optional', 'Tuple', 'List', 'get_slide_info', 'check_openslide_available', 'WSISyntheticSpec'],
    'tests/dataset_testing/unit/test_openslide_utils.py': ['MagicMock', 'WSISyntheticGenerator', 'WSISyntheticSpec', 'ErrorSimulator'],
    'tests/dataset_testing/unit/test_preprocessing.py': ['pytest', 'tempfile', 'Path', 'Mock', 'patch', 'h5py'],
    'tests/dataset_testing/validate_generators.py': ['numpy as np', 'Dict', 'Any'],
    'tests/test_flake8_lint_preservation.py': ['Path'],
}

def remove_import_line(content, import_name):
    """Remove import line containing import_name."""
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Skip lines that import the unused name
        if re.match(r'^(from .+ )?import ', line):
            if import_name in line:
                # Check if it's a multi-import line
                if ',' in line:
                    # Remove just this import from the line
                    line = re.sub(rf',?\s*{re.escape(import_name)}\s*,?', '', line)
                    line = re.sub(r',\s*,', ',', line)  # Clean up double commas
                    line = re.sub(r'import\s*,', 'import ', line)  # Clean up "import ,"
                    line = re.sub(r',\s*$', '', line)  # Clean up trailing comma
                    if 'import' in line and not re.search(r'import\s+\w', line):
                        continue  # Skip if no imports left
                else:
                    continue  # Skip entire line
        new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_file(filepath, unused_imports):
    """Remove unused imports from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        for import_name in unused_imports:
            content = remove_import_line(content, import_name)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all files with unused imports."""
    fixed = 0
    for filepath, unused in UNUSED_IMPORTS.items():
        path = Path(filepath)
        if path.exists():
            if fix_file(path, unused):
                fixed += 1
                print(f"Fixed: {filepath} (removed {len(unused)} imports)")
    
    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
