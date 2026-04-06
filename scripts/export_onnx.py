#!/usr/bin/env python3
"""
ONNX Model Export Script

This script exports trained PyTorch models to ONNX format for deployment
in production environments. ONNX provides:
- Cross-platform compatibility
- Optimized inference performance
- Support for various deployment targets (TensorRT, ONNX Runtime, etc.)

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output models/model.onnx
    
    # With custom input shapes
    python scripts/export_onnx.py \
        --checkpoint checkpoints/best_model.pth \
        --output models/model.onnx \
        --batch-size 1 \
        --wsi-size 224 \
        --num-clinical 10
        
    # With optimization
    python scripts/export_onnx.py \
        --checkpoint checkpoints/best_model.pth \
        --output models/model.onnx \
        --optimize \
        --opset-version 14
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multimodal import MultimodalFusionModel


class ONNXExporter:
    """Export PyTorch models to ONNX format."""
    
    def __init__(
        self,
        checkpoint_path: str,
        output_path: str,
        batch_size: int = 1,
        wsi_size: int = 224,
        num_clinical: int = 10,
        opset_version: int = 14,
        optimize: bool = False,
        verbose: bool = True
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            checkpoint_path: Path to PyTorch checkpoint
            output_path: Path to save ONNX model
            batch_size: Batch size for export
            wsi_size: WSI feature dimension
            num_clinical: Number of clinical features
            opset_version: ONNX opset version
            optimize: Whether to optimize the ONNX model
            verbose: Whether to print detailed information
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.wsi_size = wsi_size
        self.num_clinical = num_clinical
        self.opset_version = opset_version
        self.optimize = optimize
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> nn.Module:
        """Load PyTorch model from checkpoint."""
        if self.verbose:
            print(f"Loading checkpoint from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Use default configuration
            config = {
                'wsi_dim': self.wsi_size,
                'clinical_dim': self.num_clinical,
                'embed_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'num_classes': 4
            }
            if self.verbose:
                print("Warning: No config found in checkpoint, using defaults")
        
        # Create model
        model = MultimodalFusionModel(
            wsi_dim=config.get('wsi_dim', self.wsi_size),
            clinical_dim=config.get('clinical_dim', self.num_clinical),
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            num_classes=config.get('num_classes', 4)
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        if self.verbose:
            print(f"Model loaded successfully")
            print(f"  WSI dim: {config.get('wsi_dim', self.wsi_size)}")
            print(f"  Clinical dim: {config.get('clinical_dim', self.num_clinical)}")
            print(f"  Embed dim: {config.get('embed_dim', 256)}")
            print(f"  Num classes: {config.get('num_classes', 4)}")
        
        return model
    
    def create_dummy_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy inputs for tracing."""
        wsi_features = torch.randn(self.batch_size, self.wsi_size)
        clinical_features = torch.randn(self.batch_size, self.num_clinical)
        
        if self.verbose:
            print(f"\nDummy input shapes:")
            print(f"  WSI features: {wsi_features.shape}")
            print(f"  Clinical features: {clinical_features.shape}")
        
        return wsi_features, clinical_features
    
    def export(self) -> None:
        """Export model to ONNX format."""
        # Load model
        model = self.load_model()
        
        # Create dummy inputs
        wsi_features, clinical_features = self.create_dummy_inputs()
        
        # Export to ONNX
        if self.verbose:
            print(f"\nExporting to ONNX...")
            print(f"  Output path: {self.output_path}")
            print(f"  Opset version: {self.opset_version}")
        
        torch.onnx.export(
            model,
            (wsi_features, clinical_features),
            str(self.output_path),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['wsi_features', 'clinical_features'],
            output_names=['logits', 'embeddings'],
            dynamic_axes={
                'wsi_features': {0: 'batch_size'},
                'clinical_features': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'embeddings': {0: 'batch_size'}
            },
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"✓ Model exported successfully to {self.output_path}")
        
        # Optimize if requested
        if self.optimize:
            self.optimize_model()
        
        # Verify export
        self.verify_export()
    
    def optimize_model(self) -> None:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            if self.verbose:
                print("\nOptimizing ONNX model...")
            
            # Load ONNX model
            onnx_model = onnx.load(str(self.output_path))
            
            # Optimize
            optimized_model = optimizer.optimize_model(
                str(self.output_path),
                model_type='bert',  # Use transformer optimizations
                num_heads=8,
                hidden_size=256
            )
            
            # Save optimized model
            optimized_path = self.output_path.with_suffix('.optimized.onnx')
            optimized_model.save_model_to_file(str(optimized_path))
            
            if self.verbose:
                print(f"✓ Optimized model saved to {optimized_path}")
        
        except ImportError:
            if self.verbose:
                print("Warning: onnxruntime not installed, skipping optimization")
                print("Install with: pip install onnxruntime")
    
    def verify_export(self) -> None:
        """Verify ONNX export by running inference."""
        try:
            import onnx
            import onnxruntime as ort
            
            if self.verbose:
                print("\nVerifying ONNX export...")
            
            # Check model
            onnx_model = onnx.load(str(self.output_path))
            onnx.checker.check_model(onnx_model)
            
            if self.verbose:
                print("✓ ONNX model is valid")
            
            # Test inference
            ort_session = ort.InferenceSession(str(self.output_path))
            
            # Create dummy inputs
            wsi_features, clinical_features = self.create_dummy_inputs()
            
            # Run inference
            outputs = ort_session.run(
                None,
                {
                    'wsi_features': wsi_features.numpy(),
                    'clinical_features': clinical_features.numpy()
                }
            )
            
            if self.verbose:
                print("✓ ONNX inference successful")
                print(f"  Output shapes:")
                for i, output in enumerate(outputs):
                    print(f"    Output {i}: {output.shape}")
            
            # Get model info
            if self.verbose:
                print(f"\nModel information:")
                print(f"  File size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")
                print(f"  Inputs: {[inp.name for inp in ort_session.get_inputs()]}")
                print(f"  Outputs: {[out.name for out in ort_session.get_outputs()]}")
        
        except ImportError:
            if self.verbose:
                print("Warning: onnx or onnxruntime not installed, skipping verification")
                print("Install with: pip install onnx onnxruntime")
        except Exception as e:
            print(f"Error during verification: {e}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save ONNX model'
    )
    
    # Model configuration
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for export'
    )
    parser.add_argument(
        '--wsi-size',
        type=int,
        default=224,
        help='WSI feature dimension'
    )
    parser.add_argument(
        '--num-clinical',
        type=int,
        default=10,
        help='Number of clinical features'
    )
    
    # Export options
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize ONNX model'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create exporter
    exporter = ONNXExporter(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        wsi_size=args.wsi_size,
        num_clinical=args.num_clinical,
        opset_version=args.opset_version,
        optimize=args.optimize,
        verbose=not args.quiet
    )
    
    # Export model
    try:
        exporter.export()
        print("\n✓ Export completed successfully!")
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
