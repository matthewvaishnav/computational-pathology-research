"""
Compare baseline vs aggressive training configurations.
Shows exactly what changed and why.
"""

import yaml
from pathlib import Path

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def compare_configs():
    baseline = load_config("experiments/configs/pcam_full_20_epochs.yaml")
    aggressive = load_config("experiments/configs/pcam_aggressive.yaml")
    
    print("=" * 80)
    print("BASELINE vs AGGRESSIVE CONFIGURATION COMPARISON")
    print("=" * 80)
    print()
    
    # Model architecture
    print("📐 MODEL ARCHITECTURE")
    print("-" * 80)
    print(f"Feature Extractor:  {baseline['model']['feature_extractor']['model']:>10} → {aggressive['model']['feature_extractor']['model']}")
    print(f"Feature Dim:        {baseline['model']['feature_extractor']['feature_dim']:>10} → {aggressive['model']['feature_extractor']['feature_dim']} (4x)")
    print(f"Embed Dim:          {baseline['model']['embed_dim']:>10} → {aggressive['model']['embed_dim']} (2x)")
    print(f"Hidden Dim:         {baseline['model']['wsi']['hidden_dim']:>10} → {aggressive['model']['wsi']['hidden_dim']} (2x)")
    print(f"Attention Heads:    {baseline['model']['wsi']['num_heads']:>10} → {aggressive['model']['wsi']['num_heads']} (2x)")
    print(f"Transformer Layers: {baseline['model']['wsi']['num_layers']:>10} → {aggressive['model']['wsi']['num_layers']}")
    print(f"Pooling:            {baseline['model']['wsi']['pooling']:>10} → {aggressive['model']['wsi']['pooling']}")
    print(f"Classification:     {len(baseline['task']['classification']['hidden_dims'])}-layer → {len(aggressive['task']['classification']['hidden_dims'])}-layer")
    print()
    
    # Regularization
    print("🛡️  REGULARIZATION")
    print("-" * 80)
    print(f"Dropout:            {baseline['task']['classification']['dropout']:>10} → {aggressive['task']['classification']['dropout']}")
    print(f"Weight Decay:       {baseline['training']['weight_decay']:>10} → {aggressive['training']['weight_decay']} (10x)")
    print(f"Batch Size:         {baseline['training']['batch_size']:>10} → {aggressive['training']['batch_size']} (4x)")
    print()
    
    # Training
    print("🎯 TRAINING")
    print("-" * 80)
    print(f"Epochs:             {baseline['training']['num_epochs']:>10} → {aggressive['training']['num_epochs']}")
    print(f"Learning Rate:      {baseline['training']['learning_rate']:>10} → {aggressive['training']['learning_rate']} (3x)")
    print(f"Warmup Epochs:      {'0':>10} → {aggressive['training']['warmup_epochs']}")
    print(f"Early Stop Patience:{baseline['early_stopping']['patience']:>10} → {aggressive['early_stopping']['patience']}")
    print()
    
    # Optimizations
    print("⚡ OPTIMIZATIONS")
    print("-" * 80)
    print(f"Mixed Precision:    {str(baseline['training']['use_amp']):>10} → {str(aggressive['training']['use_amp'])}")
    print(f"Torch Compile:      {'False':>10} → {str(aggressive['training']['use_torch_compile'])}")
    print(f"Channels Last:      {'False':>10} → {str(aggressive['training']['channels_last'])}")
    print(f"Workers:            {baseline['data']['num_workers']:>10} → {aggressive['data']['num_workers']}")
    print(f"Persistent Workers: {str(baseline['data'].get('persistent_workers', False)):>10} → {str(aggressive['data']['persistent_workers'])}")
    print()
    
    # Expected impact
    print("📊 EXPECTED IMPACT")
    print("-" * 80)
    print("Current Performance:")
    print("  Test AUC:      93.71%")
    print("  Test Accuracy: 82.74%")
    print("  Test F1:       79.65%")
    print()
    print("Target Performance:")
    print("  Test AUC:      95-96% (+1.3-2.3%)")
    print("  Test Accuracy: 88-90% (+5.3-7.3%)")
    print("  Test F1:       87-89% (+7.4-9.4%)")
    print()
    print("=" * 80)

if __name__ == "__main__":
    compare_configs()
