#!/usr/bin/env python3
"""
PANDA Manifest Builder

This script builds a manifest file for the PANDA dataset by joining 
metadata from train.csv with feature file information from HDF5 files.

Usage:
    python scripts\data\build_panda_manifest.py [--labels LABELS_PATH] 
                                               [--features FEATURES_DIR] 
                                               [--out-dir OUT_DIR]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build PANDA manifest from labels and features')
    parser.add_argument(
        '--labels', 
        type=str, 
        default=r'D:\panda\train.csv',
        help='Path to train.csv labels file'
    )
    parser.add_argument(
        '--features', 
        type=str, 
        default=r'D:\panda\features_phikon',
        help='Directory containing feature .h5 files'
    )
    parser.add_argument(
        '--out-dir', 
        type=str, 
        default='results\panda_manifest',
        help='Output directory for manifest files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Inspect only the first N feature files'
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=250,
        help='Print progress every N files'
    )
    parser.add_argument(
        '--verbose-files',
        action='store_true',
        help='Print the path of each file before opening it'
    )
    return parser.parse_args()


def inspect_h5_file(file_path: Path) -> Dict[str, Any]:
    """
    Inspect an HDF5 file to extract metadata without loading full arrays.
    
    Args:
        file_path: Path to the .h5 file
        
    Returns:
        Dictionary containing file metadata or error information
    """
    result = {
        'image_id': file_path.stem,
        'feature_path': str(file_path.absolute()),
        'feature_shape': None,
        'coordinate_shape': None,
        'num_patches': None,
        'feature_dim': None,
        'file_size_bytes': file_path.stat().st_size if file_path.exists() else 0,
        'valid': False,
        'error_message': None
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if required datasets exist
            if 'features' not in f:
                result['error_message'] = 'Missing "features" dataset'
                return result
                
            if 'coordinates' not in f:
                result['error_message'] = 'Missing "coordinates" dataset'
                return result
            
            # Get shapes without loading data
            features_shape = f['features'].shape
            coordinates_shape = f['coordinates'].shape
            
            # Validate shapes
            if len(features_shape) != 2:
                result['error_message'] = f'Features must be 2D array, got shape {features_shape}'
                return result
                
            if len(coordinates_shape) != 2:
                result['error_message'] = f'Coordinates must be 2D array, got shape {coordinates_shape}'
                return result
            
            if features_shape[0] != coordinates_shape[0]:
                result['error_message'] = f'Feature and coordinate row count mismatch: {features_shape[0]} vs {coordinates_shape[0]}'
                return result
                
            if features_shape[0] == 0:
                result['error_message'] = 'Empty features array (0 patches)'
                return result
            
            # Extract metadata
            result['feature_shape'] = features_shape
            result['coordinate_shape'] = coordinates_shape
            result['num_patches'] = features_shape[0]
            result['feature_dim'] = features_shape[1]
            result['valid'] = True
            
    except Exception as e:
        result['error_message'] = f'Error reading HDF5 file: {str(e)}'
    
    return result


def build_manifest(labels_path: str, features_dir: str, out_dir: str, limit: Optional[int] = None, progress_every: int = 250, verbose_files: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Build the PANDA manifest by joining labels with feature file information.
    
    Args:
        labels_path: Path to train.csv
        features_dir: Directory containing feature .h5 files
        out_dir: Output directory for manifest files
        limit: Optional limit on number of feature files to inspect
        progress_every: Interval for printing progress
        verbose_files: Whether to print each file path before opening
        
    Returns:
        Tuple of (manifest DataFrame, summary dictionary)
    """
    # Convert to Path objects
    labels_path = Path(labels_path)
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print(f"Loading labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    print(f"Found {len(labels_df)} labels")
    
    # Scan for feature files
    print(f"Scanning for features in: {features_dir}")
    feature_files = list(features_dir.glob('*.h5'))
    
    if limit:
        print(f"Limiting inspection to first {limit} files")
        feature_files = feature_files[:limit]
        
    print(f"Found {len(feature_files)} feature files to inspect")
    
    # Inspect each feature file
    print("Inspecting feature files...")
    feature_metadata = []
    invalid_files = []
    
    start_time = time.time()
    
    for i, f_file in enumerate(feature_files, 1):
        if verbose_files:
            print(f"opening {i}/{len(feature_files)}: {f_file}", flush=True)
            
        meta = inspect_h5_file(f_file)
        feature_metadata.append(meta)
        if not meta['valid']:
            invalid_files.append((f_file.name, meta['error_message']))
            
        if i % progress_every == 0:
            print(f"inspected {i}/{len(feature_files)} files...", flush=True)
            
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_metadata)
    
    # Join with labels
    print("Joining metadata with labels...")
    manifest_df = labels_df.merge(
        features_df, 
        left_on='image_id', 
        right_on='image_id', 
        how='left',
        suffixes=('', '_feature')
    )
    
    # Calculate summary statistics
    labels_without_features = manifest_df['valid'].isna().sum()
    features_without_labels = len([f for f in feature_metadata if f['image_id'] not in labels_df['image_id'].values])
    invalid_feature_files = len([f for f in feature_metadata if not f['valid']])
    valid_feature_files = len([f for f in feature_metadata if f['valid']])
    empty_features = len([f for f in feature_metadata if f.get('num_patches') == 0])
    coordinate_mismatch = len([f for f in feature_metadata 
                              if f.get('error_message') and 'row count mismatch' in f.get('error_message', '')])
    
    files_per_sec = len(feature_files) / elapsed_time if elapsed_time > 0 else 0
    
    summary = {
        'labels_count': int(len(labels_df)),
        'feature_files_count': int(len(feature_files)),
        'manifest_rows': int(len(manifest_df)),
        'valid_feature_files': int(valid_feature_files),
        'invalid_feature_files': int(invalid_feature_files),
        'missing_features': int(labels_without_features),
        'extra_features': int(features_without_labels),
        'empty_features': int(empty_features),
        'coordinate_features_mismatch': int(coordinate_mismatch),
        'elapsed_time_seconds': float(elapsed_time),
        'files_per_second': float(files_per_sec),
        'invalid_files_details': [
            {'file': fname, 'error': err} 
            for fname, err in invalid_files
        ]
    }
    
    return manifest_df, summary


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        manifest_df, summary = build_manifest(
            args.labels,
            args.features,
            args.out_dir,
            limit=args.limit,
            progress_every=args.progress_every,
            verbose_files=args.verbose_files
        )
        
        # Write outputs
        out_dir = Path(args.out_dir)
        manifest_path = out_dir / 'panda_phikon_manifest.csv'
        summary_path = out_dir / 'panda_phikon_summary.json'
        
        print(f"\nWriting manifest to: {manifest_path}")
        manifest_df.to_csv(manifest_path, index=False)
        
        print(f"Writing summary to: {summary_path}")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("PANDA MANIFEST BUILD SUMMARY")
        print("="*50)
        print(f"Labels count: {summary['labels_count']}")
        print(f"Feature files count: {summary['feature_files_count']}")
        print(f"Manifest rows: {summary['manifest_rows']}")
        print(f"Valid feature files: {summary['valid_feature_files']}")
        print(f"Invalid feature files: {summary['invalid_feature_files']}")
        print(f"Missing features (labels without features): {summary['missing_features']}")
        print(f"Extra features (features without labels): {summary['extra_features']}")
        print(f"Empty features: {summary['empty_features']}")
        print(f"Coordinate/features mismatches: {summary['coordinate_features_mismatch']}")
        print(f"Output manifest: {manifest_path}")
        print(f"Output summary: {summary_path}")
        print("="*50)
        
        if summary['invalid_feature_files'] > 0:
            print("\nInvalid files details:")
            for invalid in summary['invalid_files_details'][:5]:  # Show first 5
                print(f"  - {invalid['file']}: {invalid['error']}")
            if len(summary['invalid_files_details']) > 5:
                print(f"  ... and {len(summary['invalid_files_details']) - 5} more")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()