"""
PANDA Feature Verification Script

Validates PANDA feature files after download:
- Checks file format (HDF5)
- Validates feature dimensions
- Checks coordinate ranges
- Verifies labels
- Reports statistics
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def verify_hdf5_file(file_path):
    """Verify a single HDF5 feature file."""
    errors = []
    warnings = []
    info = {}
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check required keys
            required_keys = ['features', 'coords']
            optional_keys = ['label', 'slide_id']
            
            for key in required_keys:
                if key not in f:
                    errors.append(f"Missing required key: {key}")
            
            if errors:
                return {'errors': errors, 'warnings': warnings, 'info': info}
            
            # Get features
            features = f['features'][:]
            coords = f['coords'][:]
            
            # Check shapes
            info['num_patches'] = features.shape[0]
            info['feature_dim'] = features.shape[1]
            info['coord_dim'] = coords.shape[1]
            
            if features.shape[0] != coords.shape[0]:
                errors.append(f"Shape mismatch: features {features.shape[0]} != coords {coords.shape[0]}")
            
            if coords.shape[1] != 2:
                errors.append(f"Coords should be 2D, got {coords.shape[1]}D")
            
            # Check feature dimension (common: 512, 1024, 2048)
            if info['feature_dim'] not in [512, 768, 1024, 2048]:
                warnings.append(f"Unusual feature dimension: {info['feature_dim']}")
            
            # Check for NaN/Inf
            if np.isnan(features).any():
                errors.append("Features contain NaN values")
            if np.isinf(features).any():
                errors.append("Features contain Inf values")
            
            if np.isnan(coords).any():
                errors.append("Coords contain NaN values")
            if np.isinf(coords).any():
                errors.append("Coords contain Inf values")
            
            # Check coordinate ranges
            coord_min = coords.min(axis=0)
            coord_max = coords.max(axis=0)
            info['coord_range'] = {
                'x': (float(coord_min[0]), float(coord_max[0])),
                'y': (float(coord_min[1]), float(coord_max[1]))
            }
            
            # Warn if coords not normalized
            if coord_max[0] > 1.5 or coord_max[1] > 1.5:
                warnings.append(f"Coords may not be normalized: max={coord_max}")
            
            # Check label if present
            if 'label' in f:
                label = f['label'][()]
                info['label'] = int(label)
                if label < 0:
                    errors.append(f"Invalid label: {label}")
            else:
                warnings.append("No label found")
            
            # Check slide_id if present
            if 'slide_id' in f:
                slide_id = f['slide_id'][()]
                if isinstance(slide_id, bytes):
                    slide_id = slide_id.decode('utf-8')
                info['slide_id'] = slide_id
            
            # Feature statistics
            info['feature_stats'] = {
                'mean': float(features.mean()),
                'std': float(features.std()),
                'min': float(features.min()),
                'max': float(features.max())
            }
            
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
    
    return {
        'errors': errors,
        'warnings': warnings,
        'info': info
    }


def verify_directory(data_dir, max_files=None):
    """Verify all HDF5 files in directory."""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    # Find all HDF5 files
    h5_files = list(data_dir.glob('*.h5')) + list(data_dir.glob('*.hdf5'))
    
    if not h5_files:
        print(f"❌ No HDF5 files found in {data_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files")
    
    if max_files:
        h5_files = h5_files[:max_files]
        print(f"Checking first {max_files} files...")
    
    # Verify each file
    results = []
    for i, file_path in enumerate(h5_files):
        print(f"\r[{i+1}/{len(h5_files)}] Checking {file_path.name}...", end='')
        result = verify_hdf5_file(file_path)
        result['file'] = file_path.name
        results.append(result)
    
    print()  # New line after progress
    
    # Summary statistics
    total_files = len(results)
    files_with_errors = sum(1 for r in results if r['errors'])
    files_with_warnings = sum(1 for r in results if r['warnings'])
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal files: {total_files}")
    print(f"✅ Valid: {total_files - files_with_errors}")
    print(f"❌ Errors: {files_with_errors}")
    print(f"⚠️  Warnings: {files_with_warnings}")
    
    # Report errors
    if files_with_errors > 0:
        print("\n" + "=" * 60)
        print("FILES WITH ERRORS")
        print("=" * 60)
        for result in results:
            if result['errors']:
                print(f"\n{result['file']}:")
                for error in result['errors']:
                    print(f"  ❌ {error}")
    
    # Report warnings
    if files_with_warnings > 0:
        print("\n" + "=" * 60)
        print("FILES WITH WARNINGS")
        print("=" * 60)
        for result in results:
            if result['warnings']:
                print(f"\n{result['file']}:")
                for warning in result['warnings']:
                    print(f"  ⚠️  {warning}")
    
    # Dataset statistics
    if results and not files_with_errors:
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        # Collect stats
        num_patches = [r['info']['num_patches'] for r in results if 'num_patches' in r['info']]
        feature_dims = [r['info']['feature_dim'] for r in results if 'feature_dim' in r['info']]
        labels = [r['info']['label'] for r in results if 'label' in r['info']]
        
        if num_patches:
            print(f"\nPatches per slide:")
            print(f"  Mean: {np.mean(num_patches):.1f}")
            print(f"  Median: {np.median(num_patches):.1f}")
            print(f"  Min: {np.min(num_patches)}")
            print(f"  Max: {np.max(num_patches)}")
        
        if feature_dims:
            unique_dims = set(feature_dims)
            print(f"\nFeature dimensions: {unique_dims}")
            if len(unique_dims) > 1:
                print("  ⚠️  Multiple feature dimensions found!")
        
        if labels:
            unique_labels = sorted(set(labels))
            print(f"\nLabels: {unique_labels}")
            print(f"Label distribution:")
            for label in unique_labels:
                count = labels.count(label)
                pct = 100 * count / len(labels)
                print(f"  Class {label}: {count} ({pct:.1f}%)")
        
        # Sample file info
        print("\n" + "=" * 60)
        print("SAMPLE FILE (first file)")
        print("=" * 60)
        sample = results[0]['info']
        print(f"\nFile: {results[0]['file']}")
        print(f"Patches: {sample.get('num_patches', 'N/A')}")
        print(f"Feature dim: {sample.get('feature_dim', 'N/A')}")
        print(f"Label: {sample.get('label', 'N/A')}")
        if 'slide_id' in sample:
            print(f"Slide ID: {sample['slide_id']}")
        
        if 'feature_stats' in sample:
            stats = sample['feature_stats']
            print(f"\nFeature statistics:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
        
        if 'coord_range' in sample:
            coord_range = sample['coord_range']
            print(f"\nCoordinate ranges:")
            print(f"  X: [{coord_range['x'][0]:.4f}, {coord_range['x'][1]:.4f}]")
            print(f"  Y: [{coord_range['y'][0]:.4f}, {coord_range['y'][1]:.4f}]")
    
    print("\n" + "=" * 60)
    
    # Return status
    if files_with_errors > 0:
        print("\n❌ Verification FAILED - fix errors before training")
        return False
    elif files_with_warnings > 0:
        print("\n⚠️  Verification passed with warnings - review before training")
        return True
    else:
        print("\n✅ Verification PASSED - ready for training!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Verify PANDA feature files')
    parser.add_argument('data_dir', type=str, help='Directory containing HDF5 feature files')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum files to check (default: all)')
    parser.add_argument('--file', type=str, default=None, help='Check single file instead of directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PANDA Feature Verification")
    print("=" * 60)
    
    if args.file:
        # Verify single file
        print(f"\nChecking file: {args.file}")
        result = verify_hdf5_file(args.file)
        
        if result['errors']:
            print("\n❌ ERRORS:")
            for error in result['errors']:
                print(f"  {error}")
        
        if result['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in result['warnings']:
                print(f"  {warning}")
        
        if result['info']:
            print("\n📊 INFO:")
            for key, value in result['info'].items():
                print(f"  {key}: {value}")
        
        success = len(result['errors']) == 0
    else:
        # Verify directory
        success = verify_directory(args.data_dir, args.max_files)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
