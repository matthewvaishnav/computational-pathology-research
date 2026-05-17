"""
BLAZING FAST PANDA feature extraction using OpenSlide.

Uses OpenSlide's multi-resolution pyramid to read from lower resolution levels.
This is the proper way to handle whole slide images.

Speed: <1 second per slide
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.panda_dataset import PANDASlideIndex

logger = logging.getLogger(__name__)


def create_feature_extractor(model_name: str, device: torch.device) -> tuple:
    """Create feature extraction model."""
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 2048
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 512
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    return model, feature_dim


def extract_slide_features_openslide(
    slide_path: Path,
    model: nn.Module,
    device: torch.device,
    level: int = 1,
    patch_size: int = 224,
    stride: int = 224,
    batch_size: int = 64,
    max_patches: int = None,
) -> tuple:
    """Extract features using OpenSlide.
    
    Args:
        slide_path: Path to WSI
        model: Feature extractor
        device: Device
        level: Pyramid level to read from (0=full res, 1=4x down, 2=16x down)
        patch_size: Patch size
        stride: Stride
        batch_size: Batch size
        max_patches: Maximum number of patches to extract per slide (None = all)
        
    Returns:
        (features, coordinates)
    """
    # Open slide
    slide = openslide.OpenSlide(str(slide_path))
    
    # Get dimensions at chosen level
    level_dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    
    # Get thumbnail for tissue detection
    thumb_size = (2048, 2048)
    thumbnail = slide.get_thumbnail(thumb_size)
    thumb_array = np.array(thumbnail.convert('RGB'))
    
    # Tissue detection
    gray = thumb_array.mean(axis=2)
    tissue_mask = (gray < 220) & (gray > 20)
    
    # Calculate scale from thumbnail to chosen level
    scale_x = level_dims[0] / thumb_array.shape[1]
    scale_y = level_dims[1] / thumb_array.shape[0]
    
    # Find tissue patches
    patches = []
    coords = []
    
    thumb_h, thumb_w = thumb_array.shape[:2]
    patch_size_thumb = int(patch_size / scale_x)
    stride_thumb = int(stride / scale_x)
    
    transform = transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for y in range(0, thumb_h - patch_size_thumb, stride_thumb):
        for x in range(0, thumb_w - patch_size_thumb, stride_thumb):
            # Check if we've reached max_patches limit
            if max_patches is not None and len(patches) >= max_patches:
                break
                
            # Check tissue
            patch_mask = tissue_mask[y:y+patch_size_thumb, x:x+patch_size_thumb]
            if patch_mask.mean() < 0.1:
                continue
            
            # Convert to level coordinates
            level_x = int(x * scale_x)
            level_y = int(y * scale_y)
            
            # Convert to level 0 coordinates for read_region
            level0_x = int(level_x * downsample)
            level0_y = int(level_y * downsample)
            
            # Read patch from chosen level
            try:
                patch = slide.read_region(
                    (level0_x, level0_y),
                    level,
                    (patch_size, patch_size)
                )
                patch = patch.convert('RGB')
                patch_tensor = transform(patch)
                patches.append(patch_tensor)
                coords.append((level0_x, level0_y))
            except Exception as e:
                continue
        
        # Break outer loop if max_patches reached
        if max_patches is not None and len(patches) >= max_patches:
            break
    
    slide.close()
    
    if len(patches) == 0:
        return None, None
    
    # Extract features in batches
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack(patches[i:i+batch_size]).to(device)
            features = model(batch)
            features = features.squeeze(-1).squeeze(-1)
            all_features.append(features.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    coords = np.array(coords)
    
    return features, coords


def save_features(
    output_path: Path,
    features: np.ndarray,
    coordinates: np.ndarray,
    slide_id: str,
):
    """Save features to HDF5 file."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset("coordinates", data=coordinates, compression="gzip")
        f.attrs["slide_id"] = slide_id
        f.attrs["num_patches"] = len(features)
        f.attrs["feature_dim"] = features.shape[1]


def main():
    parser = argparse.ArgumentParser(description="OpenSlide PANDA feature extraction")
    parser.add_argument("--data_dir", type=str, default="data/panda")
    parser.add_argument("--output_dir", type=str, default="data/panda/features")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    parser.add_argument("--level", type=int, default=1, help="Pyramid level (0=full, 1=4x down, 2=16x down)")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--max_patches", type=int, default=None, help="Maximum patches per slide (None = all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Using OpenSlide pyramid level {args.level}")
    
    # Load slide index
    data_dir = Path(args.data_dir)
    index_path = data_dir / "slide_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Slide index not found: {index_path}")
    
    slide_index = PANDASlideIndex.load(index_path)
    logger.info(f"Loaded {len(slide_index)} slides")
    
    # Create feature extractor
    model, feature_dim = create_feature_extractor(args.model, device)
    logger.info(f"Created {args.model} feature extractor (dim={feature_dim})")
    
    # Process slides
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slides_to_process = slide_index.slides
    if args.max_slides:
        slides_to_process = slides_to_process[:args.max_slides]
    
    logger.info(f"Processing {len(slides_to_process)} slides...")
    
    for slide in tqdm(slides_to_process, desc="Processing slides"):
        output_path = output_dir / f"{slide.slide_id}.h5"
        
        if output_path.exists():
            continue
        
        slide_path = Path(slide.file_path)
        if not slide_path.exists():
            logger.warning(f"Slide not found: {slide_path}")
            continue
        
        try:
            features, coords = extract_slide_features_openslide(
                slide_path=slide_path,
                model=model,
                device=device,
                level=args.level,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
            )
            
            if features is None:
                logger.warning(f"No tissue found in {slide.slide_id}")
                continue
            
            save_features(output_path, features, coords, slide.slide_id)
            
        except Exception as e:
            logger.error(f"Error processing {slide.slide_id}: {e}")
            continue
    
    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
