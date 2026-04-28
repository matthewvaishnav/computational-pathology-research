import torch

# Check both checkpoints
checkpoints = [
    "checkpoints/pcam_real/best_model.pth",
    "checkpoints/pcam_real/pcam-1776261028_epoch_10.pth"
]

for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Epoch: {ckpt['epoch']}")
    print(f"Batch: {ckpt.get('batch_idx', 'N/A')}")
    
    if 'metrics' in ckpt:
        metrics = ckpt['metrics']
        print(f"Metrics: {metrics}")
    
    if 'best_val_auc' in ckpt:
        print(f"Best Val AUC: {ckpt['best_val_auc']}")
    
    print(f"Keys: {list(ckpt.keys())}")
