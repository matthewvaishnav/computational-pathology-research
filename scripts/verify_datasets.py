#!/usr/bin/env python3
"""
Verify downloaded datasets and generate statistics.

This script checks that datasets are properly downloaded and structured,
and generates summary statistics.
"""

import json
from pathlib import Path
import argparse
from typing import Dict, List
from collections import defaultdict


def count_files_by_extension(directory: Path) -> Dict[str, int]:
    """Count files by extension in a directory."""
    counts = defaultdict(int)
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            counts[ext] += 1
    
    return dict(counts)


def verify_lc25000(data_root: Path) -> Dict:
    """Verify LC25000 dataset."""
    dataset_dir = data_root / "multi_disease" / "lung" / "lc25000"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Count images
    file_counts = count_files_by_extension(dataset_dir)
    image_count = sum(
        count for ext, count in file_counts.items()
        if ext in ['.jpg', '.jpeg', '.png']
    )
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_images": image_count,
        "file_types": file_counts,
        "expected": 25000,
        "complete": image_count >= 20000  # Allow some tolerance
    }


def verify_nct_crc(data_root: Path) -> Dict:
    """Verify NCT-CRC-HE-100K dataset."""
    dataset_dir = data_root / "multi_disease" / "colon" / "nct_crc"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Count images
    file_counts = count_files_by_extension(dataset_dir)
    image_count = sum(
        count for ext, count in file_counts.items()
        if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    )
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_images": image_count,
        "file_types": file_counts,
        "expected": 100000,
        "complete": image_count >= 90000
    }


def verify_crc_val(data_root: Path) -> Dict:
    """Verify CRC-VAL-HE-7K dataset."""
    dataset_dir = data_root / "multi_disease" / "colon" / "crc_val"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Count images
    file_counts = count_files_by_extension(dataset_dir)
    image_count = sum(
        count for ext, count in file_counts.items()
        if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    )
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_images": image_count,
        "file_types": file_counts,
        "expected": 7180,
        "complete": image_count >= 6500
    }


def verify_ham10000(data_root: Path) -> Dict:
    """Verify HAM10000 dataset."""
    dataset_dir = data_root / "multi_disease" / "melanoma" / "ham10000"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Count images
    file_counts = count_files_by_extension(dataset_dir)
    image_count = sum(
        count for ext, count in file_counts.items()
        if ext in ['.jpg', '.jpeg', '.png']
    )
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_images": image_count,
        "file_types": file_counts,
        "expected": 10015,
        "complete": image_count >= 9000
    }


def verify_panda(data_root: Path) -> Dict:
    """Verify PANDA dataset."""
    dataset_dir = data_root / "multi_disease" / "prostate" / "panda"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Count images (WSI files)
    file_counts = count_files_by_extension(dataset_dir)
    wsi_count = sum(
        count for ext, count in file_counts.items()
        if ext in ['.tiff', '.tif', '.svs']
    )
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_slides": wsi_count,
        "file_types": file_counts,
        "expected": 10616,
        "complete": wsi_count >= 9000
    }


def verify_sicap(data_root: Path) -> Dict:
    """Verify SICAPv2 dataset."""
    dataset_dir = data_root / "multi_disease" / "prostate" / "sicap"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Count images
    file_counts = count_files_by_extension(dataset_dir)
    image_count = sum(
        count for ext, count in file_counts.items()
        if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    )
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_patches": image_count,
        "file_types": file_counts,
        "expected": 18783,
        "complete": image_count >= 17000
    }


def verify_pcam(data_root: Path) -> Dict:
    """Verify PatchCamelyon dataset."""
    dataset_dir = data_root / "pcam_real"
    
    if not dataset_dir.exists():
        return {"status": "not_found", "path": str(dataset_dir)}
    
    # Check for h5 files
    h5_files = list(dataset_dir.glob("*.h5"))
    
    return {
        "status": "found",
        "path": str(dataset_dir),
        "num_files": len(h5_files),
        "files": [f.name for f in h5_files],
        "expected": 6,  # train_x, train_y, valid_x, valid_y, test_x, test_y
        "complete": len(h5_files) >= 6
    }


def verify_vision_language(data_root: Path) -> Dict:
    """Verify vision-language datasets."""
    vl_dir = data_root / "vision_language"
    
    if not vl_dir.exists():
        return {"status": "not_found", "path": str(vl_dir)}
    
    # Check subdirectories
    subdirs = [d for d in vl_dir.iterdir() if d.is_dir()]
    
    total_pairs = 0
    datasets = {}
    
    for subdir in subdirs:
        images_dir = subdir / "images"
        captions_dir = subdir / "captions"
        
        if images_dir.exists() and captions_dir.exists():
            num_images = len(list(images_dir.glob("*.[jp][pn]g")))
            num_captions = len(list(captions_dir.glob("*.txt")))
            num_pairs = min(num_images, num_captions)
            
            datasets[subdir.name] = {
                "num_images": num_images,
                "num_captions": num_captions,
                "num_pairs": num_pairs
            }
            
            total_pairs += num_pairs
    
    return {
        "status": "found" if datasets else "empty",
        "path": str(vl_dir),
        "datasets": datasets,
        "total_pairs": total_pairs
    }


def generate_report(data_root: Path) -> Dict:
    """Generate comprehensive verification report."""
    
    print("=" * 60)
    print("Dataset Verification Report")
    print("=" * 60)
    
    report = {
        "multi_disease": {
            "breast": {
                "pcam": verify_pcam(data_root)
            },
            "lung": {
                "lc25000": verify_lc25000(data_root)
            },
            "colon": {
                "nct_crc": verify_nct_crc(data_root),
                "crc_val": verify_crc_val(data_root)
            },
            "melanoma": {
                "ham10000": verify_ham10000(data_root)
            },
            "prostate": {
                "panda": verify_panda(data_root),
                "sicap": verify_sicap(data_root)
            }
        },
        "vision_language": verify_vision_language(data_root)
    }
    
    # Print summary
    print("\n📊 Multi-Disease Datasets:")
    print("-" * 60)
    
    for disease, datasets in report["multi_disease"].items():
        print(f"\n{disease.upper()}:")
        for dataset_name, info in datasets.items():
            status_icon = "✓" if info.get("complete") else "⚠" if info["status"] == "found" else "✗"
            print(f"  {status_icon} {dataset_name}: {info['status']}")
            
            if info["status"] == "found":
                if "num_images" in info:
                    print(f"      Images: {info['num_images']:,} / {info['expected']:,}")
                elif "num_slides" in info:
                    print(f"      Slides: {info['num_slides']:,} / {info['expected']:,}")
                elif "num_patches" in info:
                    print(f"      Patches: {info['num_patches']:,} / {info['expected']:,}")
                elif "num_files" in info:
                    print(f"      Files: {info['num_files']} / {info['expected']}")
    
    print("\n\n🔬 Vision-Language Datasets:")
    print("-" * 60)
    
    vl_info = report["vision_language"]
    if vl_info["status"] == "found":
        for dataset_name, info in vl_info["datasets"].items():
            print(f"  ✓ {dataset_name}: {info['num_pairs']:,} image-text pairs")
        print(f"\n  Total: {vl_info['total_pairs']:,} pairs")
    else:
        print(f"  ✗ No vision-language datasets found")
    
    # Summary statistics
    print("\n\n📈 Summary:")
    print("-" * 60)
    
    total_datasets = 0
    complete_datasets = 0
    partial_datasets = 0
    missing_datasets = 0
    
    for disease, datasets in report["multi_disease"].items():
        for dataset_name, info in datasets.items():
            total_datasets += 1
            if info.get("complete"):
                complete_datasets += 1
            elif info["status"] == "found":
                partial_datasets += 1
            else:
                missing_datasets += 1
    
    print(f"Total datasets: {total_datasets}")
    print(f"  ✓ Complete: {complete_datasets}")
    print(f"  ⚠ Partial: {partial_datasets}")
    print(f"  ✗ Missing: {missing_datasets}")
    
    # Readiness assessment
    print("\n\n🎯 Training Readiness:")
    print("-" * 60)
    
    if complete_datasets >= 3:
        print("✓ Ready for multi-disease training")
        print(f"  {complete_datasets} complete datasets available")
    elif complete_datasets >= 1:
        print("⚠ Limited multi-disease training possible")
        print(f"  Only {complete_datasets} complete dataset(s)")
        print("  Recommend downloading more datasets")
    else:
        print("✗ Not ready for training")
        print("  No complete datasets found")
        print("  Run: python scripts/download_public_datasets.py")
    
    if vl_info["status"] == "found" and vl_info["total_pairs"] >= 10000:
        print("\n✓ Ready for vision-language training")
        print(f"  {vl_info['total_pairs']:,} image-text pairs available")
    elif vl_info["status"] == "found":
        print("\n⚠ Limited vision-language training possible")
        print(f"  Only {vl_info['total_pairs']:,} pairs (recommend 100K+)")
    else:
        print("\n✗ Not ready for vision-language training")
        print("  No vision-language datasets found")
        print("  Run: python scripts/generate_captions_gpt4v.py")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify downloaded datasets"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for datasets (default: data/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for report"
    )
    
    args = parser.parse_args()
    
    # Generate report
    report = generate_report(args.data_root)
    
    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
