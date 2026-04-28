import torch

# Check the model config stored in the checkpoint
ckpt_path = "checkpoints/pcam_real/pcam-1776261028_epoch_10.pth"
ckpt = torch.load(ckpt_path, map_location='cpu')

print("Model config from checkpoint:")
print(f"embed_dim: {ckpt['config']['model']['embed_dim']}")
print(f"wsi.input_dim: {ckpt['config']['model']['wsi']['input_dim']}")
print(f"wsi.hidden_dim: {ckpt['config']['model']['wsi']['hidden_dim']}")
print(f"wsi.num_heads: {ckpt['config']['model']['wsi']['num_heads']}")
print(f"wsi.num_layers: {ckpt['config']['model']['wsi']['num_layers']}")
print(f"wsi.pooling: {ckpt['config']['model']['wsi']['pooling']}")
