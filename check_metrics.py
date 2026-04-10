import torch
import json
from pathlib import Path

ckpt_path = Path("checkpoints/pcam_real/best_model.pth")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("=== Checkpoint Metrics ===")
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"\nMetrics:")
metrics = ckpt.get("metrics", {})
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print(f"\nConfig keys: {list(ckpt.get('config', {}).keys())}")
