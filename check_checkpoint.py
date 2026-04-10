import torch
from pathlib import Path

ckpt_path = Path("checkpoints/pcam_real/best_model.pth")
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint found: {ckpt_path}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    print(
        f"Val Accuracy: {ckpt.get('val_accuracy', 'N/A'):.4f}"
        if "val_accuracy" in ckpt
        else "Val Accuracy: N/A"
    )
    print(f"Val AUC: {ckpt.get('val_auc', 'N/A'):.4f}" if "val_auc" in ckpt else "Val AUC: N/A")
    print(f"Available keys: {list(ckpt.keys())}")
else:
    print(f"Checkpoint not found: {ckpt_path}")
