import torch

# Check the config stored in the checkpoint
ckpt_path = "checkpoints/pcam_real/pcam-1776261028_epoch_10.pth"
ckpt = torch.load(ckpt_path, map_location='cpu')

print("Config from checkpoint:")
print(f"Data root_dir: {ckpt['config']['data']['root_dir']}")
print(f"Checkpoint dir: {ckpt['config']['checkpoint']['checkpoint_dir']}")
print(f"Log dir: {ckpt['config']['logging']['log_dir']}")
