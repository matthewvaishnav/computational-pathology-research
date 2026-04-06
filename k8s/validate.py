#!/usr/bin/env python3
"""Validate Kubernetes YAML files."""

import yaml
from pathlib import Path

def validate_yaml_files():
    """Validate all YAML files in k8s directory."""
    k8s_dir = Path('k8s')
    yaml_files = list(k8s_dir.glob('*.yaml'))
    
    valid_count = 0
    errors = []
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file) as f:
                # Load all documents in the file
                docs = list(yaml.safe_load_all(f))
                valid_count += 1
                print(f'✓ {yaml_file.name} ({len(docs)} document(s))')
        except Exception as e:
            errors.append(f'✗ {yaml_file.name}: {e}')
            print(f'✗ {yaml_file.name}: {e}')
    
    print(f'\nValidation complete: {valid_count}/{len(yaml_files)} files valid')
    
    if errors:
        print('\nErrors:')
        for error in errors:
            print(f'  {error}')
        return False
    
    return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if validate_yaml_files() else 1)
