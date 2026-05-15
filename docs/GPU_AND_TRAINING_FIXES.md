# GPU and Training Issues - Diagnosis and Fixes

## Issues Identified

### 1. GPU Not Being Used (CRITICAL)
**Symptom**: Log shows "Using device: cpu" despite having RTX 4070 GPU

**Possible Causes**:
- PyTorch CPU-only version installed (most likely)
- CUDA drivers not installed or outdated
- CUDA version mismatch between PyTorch and drivers
- PyTorch/CUDA initialization hanging

**Diagnosis**:
Run the diagnostic script (without hanging):
```bash
python check_gpu.py
```

**Fix Options**:

A. **If PyTorch is CPU-only** (most likely):
```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version (for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

B. **If CUDA drivers need update**:
- Download latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
- Select RTX 4070, Windows, and download
- Install and reboot

C. **If PyTorch hangs on CUDA check**:
- This indicates a driver/CUDA mismatch
- Reinstall both NVIDIA drivers and PyTorch with matching CUDA version

### 2. Training Loss Error (FIXED)
**Symptom**: `Target size (torch.Size([4])) must be the same as input size (torch.Size([4, 2]))`

**Root Cause**: 
- `BCEWithLogitsLoss` expects single logits `[batch]` for binary classification
- Mean/max pooling models return 2-class logits `[batch, 2]`
- Mismatch between loss function and model output shape

**Fix Applied**:
Changed `experiments/compare_attention_models.py` to use `CrossEntropyLoss` instead:
- Line ~207: Changed criterion to `nn.CrossEntropyLoss()`
- Line ~215: Changed to `criterion(logits, labels.long())`
- Line ~241: Changed to use `torch.softmax(logits, dim=-1)[:, 1]` for probabilities
- Line ~285: Same softmax change for test evaluation

### 3. Extremely Slow Data Loading
**Symptom**: 20+ minutes between log entries during data loading

**Possible Causes**:
- HDF5 files being loaded from slow storage (HDD instead of SSD)
- Too many workers causing I/O bottleneck
- Large slide files with many patches
- CPU-only mode making everything slower

**Fixes**:
1. **Reduce num_workers** in `experiments/configs/comparison.yaml`:
   ```yaml
   data:
     num_workers: 0  # or 1 for single worker
   ```

2. **Move data to SSD** if currently on HDD

3. **Reduce max_patches_per_slide** for faster loading:
   ```yaml
   data:
     slide:
       max_patches_per_slide: 500  # Reduced from 1000
   ```

4. **Enable GPU** - This will speed up everything significantly

## Recommended Action Plan

1. **First Priority - Fix GPU**:
   ```bash
   # Check current PyTorch installation
   python check_gpu.py
   
   # If CPU-only, reinstall with CUDA support
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Verify GPU is detected
   python check_gpu.py
   ```

2. **Second Priority - Test Training Fix**:
   ```bash
   # The loss error is already fixed in compare_attention_models.py
   # Test with small dataset first
   python experiments/compare_attention_models.py --config experiments/configs/comparison.yaml --models mean
   ```

3. **Third Priority - Optimize Data Loading**:
   - Edit `experiments/configs/comparison.yaml`
   - Set `num_workers: 0` or `1`
   - Reduce `max_patches_per_slide` to 500

## Expected Performance After Fixes

With GPU enabled:
- Data loading: ~1-2 minutes (vs 20+ minutes on CPU)
- Training epoch: ~5-10 minutes per epoch (vs hours on CPU)
- Total comparison time: ~1-2 hours for all models (vs days on CPU)

Without GPU (CPU only):
- Training will be 10-50x slower
- Not recommended for full experiments
- Consider using smaller synthetic dataset for testing

## Verification

After applying fixes, you should see:
```
Using device: cuda
CUDA device name: NVIDIA GeForce RTX 4070
```

And training should proceed without the target size error.
