"""
Batch inference script for multiple WSI slides using TorchScript model.

Optimized for production deployment with:
- TorchScript model loading (2-3x faster)
- Batch processing for efficiency
- Progress tracking and error handling
- Result aggregation and export

Usage:
    python scripts/batch_inference.py \
        --model models/model_scripted.pt \
        --input-dir data/slides/ \
        --output results/predictions.csv \
        --batch-size 32
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_scripted_model(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load TorchScript model."""
    logger.info(f"Loading TorchScript model from {model_path}")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def load_slide_features(slide_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load preprocessed slide features."""
    # Placeholder - implement based on your feature extraction pipeline
    # This should load WSI features, genomic data, clinical text, etc.
    
    # Example structure:
    features = torch.load(slide_path, map_location=device)
    
    return {
        'wsi_features': features.get('wsi_features'),
        'genomic': features.get('genomic'),
        'clinical_text': features.get('clinical_text'),
        'wsi_mask': features.get('wsi_mask'),
        'clinical_mask': features.get('clinical_mask'),
    }


def batch_inference(
    model: torch.jit.ScriptModule,
    slide_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict]:
    """Run batch inference on multiple slides."""
    results = []
    
    logger.info(f"Processing {len(slide_paths)} slides with batch_size={batch_size}")
    
    # Process in batches
    for i in tqdm(range(0, len(slide_paths), batch_size), desc="Inference"):
        batch_paths = slide_paths[i:i + batch_size]
        
        # Load features for batch
        batch_features = []
        valid_paths = []
        
        for slide_path in batch_paths:
            try:
                features = load_slide_features(slide_path, device)
                batch_features.append(features)
                valid_paths.append(slide_path)
            except Exception as e:
                logger.warning(f"Failed to load {slide_path}: {e}")
                continue
        
        if not batch_features:
            continue
        
        # Stack features into batch tensors
        batch_wsi = None
        batch_genomic = None
        batch_clinical = None
        batch_wsi_mask = None
        batch_clinical_mask = None
        
        if batch_features[0]['wsi_features'] is not None:
            batch_wsi = torch.stack([f['wsi_features'] for f in batch_features])
            if batch_features[0]['wsi_mask'] is not None:
                batch_wsi_mask = torch.stack([f['wsi_mask'] for f in batch_features])
        
        if batch_features[0]['genomic'] is not None:
            batch_genomic = torch.stack([f['genomic'] for f in batch_features])
        
        if batch_features[0]['clinical_text'] is not None:
            batch_clinical = torch.stack([f['clinical_text'] for f in batch_features])
            if batch_features[0]['clinical_mask'] is not None:
                batch_clinical_mask = torch.stack([f['clinical_mask'] for f in batch_features])
        
        # Run inference
        with torch.no_grad():
            logits = model(
                batch_wsi,
                batch_genomic,
                batch_clinical,
                batch_wsi_mask,
                batch_clinical_mask,
            )
            
            # Get predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        
        # Store results
        for j, slide_path in enumerate(valid_paths):
            results.append({
                'slide_id': slide_path.stem,
                'slide_path': str(slide_path),
                'prediction': preds[j].item(),
                'confidence': probs[j].max().item(),
                'probabilities': probs[j].cpu().numpy().tolist(),
            })
    
    return results


def save_results(results: List[Dict], output_path: Path) -> None:
    """Save inference results to CSV."""
    logger.info(f"Saving results to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        if not results:
            logger.warning("No results to save")
            return
        
        fieldnames = ['slide_id', 'slide_path', 'prediction', 'confidence']
        
        # Add probability columns
        num_classes = len(results[0]['probabilities'])
        for i in range(num_classes):
            fieldnames.append(f'prob_class_{i}')
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'slide_id': result['slide_id'],
                'slide_path': result['slide_path'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
            }
            
            # Add probabilities
            for i, prob in enumerate(result['probabilities']):
                row[f'prob_class_{i}'] = f"{prob:.6f}"
            
            writer.writerow(row)
    
    logger.info(f"✓ Saved {len(results)} results")


def main():
    parser = argparse.ArgumentParser(description="Batch inference with TorchScript model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to TorchScript model (.pt file)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed slide features",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions.csv",
        help="Output CSV file for predictions",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="File pattern for slide features",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    model = load_scripted_model(model_path, device)
    
    # Find slide features
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    slide_paths = sorted(input_dir.glob(args.pattern))
    if not slide_paths:
        logger.error(f"No slides found matching pattern '{args.pattern}' in {input_dir}")
        return
    
    logger.info(f"Found {len(slide_paths)} slides")
    
    # Run batch inference
    results = batch_inference(model, slide_paths, device, args.batch_size)
    
    # Save results
    output_path = Path(args.output)
    save_results(results, output_path)
    
    # Summary statistics
    if results:
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        logger.info(f"\n{'='*60}")
        logger.info("Inference Summary:")
        logger.info(f"{'='*60}")
        logger.info(f"Total slides processed: {len(results)}")
        logger.info(f"Average confidence: {sum(confidences)/len(confidences):.4f}")
        
        # Class distribution
        from collections import Counter
        class_counts = Counter(predictions)
        logger.info(f"\nClass distribution:")
        for cls, count in sorted(class_counts.items()):
            pct = count / len(results) * 100
            logger.info(f"  Class {cls}: {count} ({pct:.1f}%)")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
