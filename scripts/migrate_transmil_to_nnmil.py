#!/usr/bin/env python3
"""
Migration script for converting TransMIL checkpoints to nnMIL format.

This script loads a TransMIL checkpoint and converts it to nnMIL format,
transferring compatible weights and creating appropriate metadata.

Usage:
    python scripts/migrate_transmil_to_nnmil.py \
        --input checkpoints/transmil_best.pth \
        --output checkpoints/nnmil_migrated.pth \
        --config configs/disease_subtyping.yaml

Migration Process:
1. Load TransMIL checkpoint
2. Extract compatible weights (feature projection, classifier)
3. Create nnMIL model with equivalent architecture
4. Transfer weights to nnMIL model
5. Save as nnMIL checkpoint with metadata

Compatible Components:
- Feature projection layers (if dimensions match)
- Classifier head (if num_classes match)
- Attention mechanism (with adaptation)

Non-transferable Components:
- Multi-scale fusion layers (nnMIL-specific)
- Uncertainty estimation layers (nnMIL-specific)
- Task-aware batch sampler state (different architecture)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.nnmil_config import nnMILConfig
from models import nnMIL, TransMIL


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_transmil_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load TransMIL checkpoint.
    
    Args:
        checkpoint_path: Path to TransMIL checkpoint
    
    Returns:
        Checkpoint dictionary
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TransMIL checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate checkpoint format
    required_keys = ['model_state_dict']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Invalid TransMIL checkpoint: missing '{key}'")
    
    return checkpoint


def analyze_transmil_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Analyze TransMIL architecture from state dict.
    
    Args:
        state_dict: TransMIL model state dict
    
    Returns:
        Architecture information
    """
    arch_info = {}
    
    # Extract dimensions from weight shapes
    for name, tensor in state_dict.items():
        if 'feature_projection' in name and 'weight' in name:
            if tensor.dim() == 2:
                arch_info['feature_dim'] = tensor.shape[1]
                arch_info['hidden_dim'] = tensor.shape[0]
        
        elif 'classifier' in name and 'weight' in name:
            if tensor.dim() == 2:
                arch_info['num_classes'] = tensor.shape[0]
                if 'hidden_dim' not in arch_info:
                    arch_info['hidden_dim'] = tensor.shape[1]
        
        elif 'attention' in name and 'weight' in name:
            arch_info['has_attention'] = True
    
    # Set defaults if not found
    arch_info.setdefault('feature_dim', 1024)
    arch_info.setdefault('hidden_dim', 256)
    arch_info.setdefault('num_classes', 2)
    arch_info.setdefault('has_attention', True)
    
    return arch_info


def create_weight_mapping(
    transmil_state: Dict[str, torch.Tensor],
    nnmil_state: Dict[str, torch.Tensor]
) -> Dict[str, str]:
    """
    Create mapping between TransMIL and nnMIL parameter names.
    
    Args:
        transmil_state: TransMIL state dict
        nnmil_state: nnMIL state dict
    
    Returns:
        Dictionary mapping TransMIL names to nnMIL names
    """
    mapping = {}
    
    # Direct mappings for compatible layers
    direct_mappings = {
        # Feature projection
        'feature_projection.weight': 'feature_projection.weight',
        'feature_projection.bias': 'feature_projection.bias',
        
        # Attention mechanism
        'attention.V.weight': 'attention.V.weight',
        'attention.V.bias': 'attention.V.bias',
        'attention.U.weight': 'attention.U.weight',
        'attention.U.bias': 'attention.U.bias',
        'attention.w.weight': 'attention.w.weight',
        'attention.w.bias': 'attention.w.bias',
        
        # Classifier
        'classifier.0.weight': 'classifier.0.weight',
        'classifier.0.bias': 'classifier.0.bias',
        'classifier.2.weight': 'classifier.2.weight',
        'classifier.2.bias': 'classifier.2.bias',
    }
    
    # Check which mappings are valid
    for transmil_name, nnmil_name in direct_mappings.items():
        if transmil_name in transmil_state and nnmil_name in nnmil_state:
            # Verify shapes match
            transmil_shape = transmil_state[transmil_name].shape
            nnmil_shape = nnmil_state[nnmil_name].shape
            
            if transmil_shape == nnmil_shape:
                mapping[transmil_name] = nnmil_name
            else:
                logging.warning(
                    f"Shape mismatch for {transmil_name}: "
                    f"TransMIL {transmil_shape} vs nnMIL {nnmil_shape}"
                )
    
    return mapping


def transfer_weights(
    transmil_checkpoint: Dict[str, Any],
    nnmil_model: nn.Module,
    weight_mapping: Dict[str, str]
) -> Tuple[int, int]:
    """
    Transfer weights from TransMIL to nnMIL model.
    
    Args:
        transmil_checkpoint: TransMIL checkpoint
        nnmil_model: nnMIL model instance
        weight_mapping: Parameter name mapping
    
    Returns:
        Tuple of (transferred_params, total_params)
    """
    transmil_state = transmil_checkpoint['model_state_dict']
    nnmil_state = nnmil_model.state_dict()
    
    transferred_params = 0
    total_params = len(nnmil_state)
    
    # Transfer mapped weights
    for transmil_name, nnmil_name in weight_mapping.items():
        nnmil_state[nnmil_name] = transmil_state[transmil_name].clone()
        transferred_params += 1
        logging.info(f"Transferred: {transmil_name} -> {nnmil_name}")
    
    # Load the updated state dict
    nnmil_model.load_state_dict(nnmil_state)
    
    return transferred_params, total_params


def create_nnmil_checkpoint(
    nnmil_model: nn.Module,
    nnmil_config: nnMILConfig,
    transmil_checkpoint: Dict[str, Any],
    migration_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create nnMIL checkpoint with migrated weights and metadata.
    
    Args:
        nnmil_model: nnMIL model with transferred weights
        nnmil_config: nnMIL configuration
        transmil_checkpoint: Original TransMIL checkpoint
        migration_info: Migration metadata
    
    Returns:
        nnMIL checkpoint dictionary
    """
    checkpoint = {
        'model_state_dict': nnmil_model.state_dict(),
        'config': nnmil_config.to_dict(),
        'epoch': transmil_checkpoint.get('epoch', 0),
        'best_metric': transmil_checkpoint.get('best_metric', 0.0),
        'migration_info': migration_info,
        'model_type': 'nnmil',
        'migrated_from': 'transmil',
        'migration_timestamp': torch.tensor(torch.get_default_dtype()).new_tensor([0]).item()  # Placeholder
    }
    
    # Copy optimizer state if available (may not be compatible)
    if 'optimizer_state_dict' in transmil_checkpoint:
        logging.warning(
            "Optimizer state from TransMIL may not be compatible with nnMIL. "
            "Consider restarting training with fresh optimizer."
        )
        checkpoint['original_optimizer_state_dict'] = transmil_checkpoint['optimizer_state_dict']
    
    # Copy training history if available
    if 'training_history' in transmil_checkpoint:
        checkpoint['original_training_history'] = transmil_checkpoint['training_history']
    
    return checkpoint


def migrate_transmil_to_nnmil(
    input_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
    force_overwrite: bool = False
) -> Dict[str, Any]:
    """
    Main migration function.
    
    Args:
        input_path: Path to TransMIL checkpoint
        output_path: Path for nnMIL checkpoint
        config_path: Path to nnMIL config (optional)
        force_overwrite: Whether to overwrite existing output
    
    Returns:
        Migration summary
    """
    logger = logging.getLogger(__name__)
    
    # Check output path
    if output_path.exists() and not force_overwrite:
        raise FileExistsError(
            f"Output file exists: {output_path}. Use --force to overwrite."
        )
    
    # Load TransMIL checkpoint
    logger.info(f"Loading TransMIL checkpoint: {input_path}")
    transmil_checkpoint = load_transmil_checkpoint(input_path)
    
    # Analyze TransMIL architecture
    logger.info("Analyzing TransMIL architecture...")
    arch_info = analyze_transmil_architecture(transmil_checkpoint['model_state_dict'])
    logger.info(f"Detected architecture: {arch_info}")
    
    # Load or create nnMIL config
    if config_path is not None:
        logger.info(f"Loading nnMIL config: {config_path}")
        nnmil_config = nnMILConfig.from_yaml(config_path)
        
        # Validate compatibility
        if nnmil_config.feature_dim != arch_info['feature_dim']:
            logger.warning(
                f"Feature dimension mismatch: config {nnmil_config.feature_dim} "
                f"vs TransMIL {arch_info['feature_dim']}"
            )
        
        if nnmil_config.num_classes != arch_info['num_classes']:
            logger.warning(
                f"Number of classes mismatch: config {nnmil_config.num_classes} "
                f"vs TransMIL {arch_info['num_classes']}"
            )
    
    else:
        logger.info("Creating nnMIL config from TransMIL architecture...")
        nnmil_config = nnMILConfig(
            feature_dim=arch_info['feature_dim'],
            hidden_dim=arch_info['hidden_dim'],
            num_classes=arch_info['num_classes'],
            dropout=0.25,  # Default
            multi_scale=False,  # TransMIL is single-scale
            task_type='classification'  # Default
        )
    
    # Create nnMIL model
    logger.info("Creating nnMIL model...")
    nnmil_model = nnMIL(
        feature_dim=nnmil_config.feature_dim,
        hidden_dim=nnmil_config.hidden_dim,
        num_classes=nnmil_config.num_classes,
        dropout=nnmil_config.dropout,
        multi_scale=nnmil_config.multi_scale,
        fusion_type=nnmil_config.fusion_type
    )
    
    # Create weight mapping
    logger.info("Creating weight mapping...")
    weight_mapping = create_weight_mapping(
        transmil_checkpoint['model_state_dict'],
        nnmil_model.state_dict()
    )
    logger.info(f"Found {len(weight_mapping)} compatible parameter mappings")
    
    # Transfer weights
    logger.info("Transferring weights...")
    transferred_params, total_params = transfer_weights(
        transmil_checkpoint, nnmil_model, weight_mapping
    )
    
    transfer_ratio = transferred_params / total_params
    logger.info(
        f"Transferred {transferred_params}/{total_params} parameters "
        f"({transfer_ratio:.1%})"
    )
    
    # Create migration info
    migration_info = {
        'source_checkpoint': str(input_path),
        'source_architecture': arch_info,
        'weight_mapping': weight_mapping,
        'transferred_parameters': transferred_params,
        'total_parameters': total_params,
        'transfer_ratio': transfer_ratio,
        'compatible_components': list(weight_mapping.keys()),
        'migration_notes': [
            "Multi-scale fusion layers initialized randomly (nnMIL-specific)",
            "Uncertainty estimation layers initialized randomly (nnMIL-specific)",
            "Optimizer state not transferred (architecture differences)",
        ]
    }
    
    # Create nnMIL checkpoint
    logger.info("Creating nnMIL checkpoint...")
    nnmil_checkpoint = create_nnmil_checkpoint(
        nnmil_model, nnmil_config, transmil_checkpoint, migration_info
    )
    
    # Save checkpoint
    logger.info(f"Saving nnMIL checkpoint: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(nnmil_checkpoint, output_path)
    
    # Create migration summary
    summary = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'config_path': str(config_path) if config_path else None,
        'migration_info': migration_info,
        'success': True
    }
    
    logger.info("Migration completed successfully!")
    logger.info(f"Summary: {transfer_ratio:.1%} of parameters transferred")
    
    return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate TransMIL checkpoint to nnMIL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Path to TransMIL checkpoint file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Path for output nnMIL checkpoint file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to nnMIL configuration file (optional)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing output file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    try:
        # Run migration
        summary = migrate_transmil_to_nnmil(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            force_overwrite=args.force
        )
        
        # Print summary
        print("\nMigration Summary:")
        print("=" * 50)
        print(f"Input:  {summary['input_path']}")
        print(f"Output: {summary['output_path']}")
        print(f"Config: {summary['config_path'] or 'Auto-generated'}")
        print(f"Transfer Ratio: {summary['migration_info']['transfer_ratio']:.1%}")
        print(f"Transferred: {summary['migration_info']['transferred_parameters']} parameters")
        print(f"Total: {summary['migration_info']['total_parameters']} parameters")
        
        print("\nCompatible Components:")
        for component in summary['migration_info']['compatible_components']:
            print(f"  ✓ {component}")
        
        print("\nMigration Notes:")
        for note in summary['migration_info']['migration_notes']:
            print(f"  • {note}")
        
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())