# Transfer Checklist for Other PC

## Files to Transfer

### 1. Code (via Git)
```bash
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research
git pull origin main
```

All code is in repo. Latest commit: `588012ee`

### 2. Data Files (Manual Transfer)

**Required:**
- `panda/features_resnet50_300patches/*.h5` (1,365 files, ~2-3GB total)
- `panda/splits.json` (train/val/test splits)
- `panda/train.csv` (labels reference)

**Transfer methods:**
- Google Drive (already used for features)
- USB drive
- Network share
- rsync/scp

**Verify after transfer:**
```bash
python scripts/verify_panda_features.py panda/features_resnet50_300patches --max_files 10
```

Should show:
- ✅ 1,365 files
- ✅ Labels present
- ✅ Coords normalized [0,1]
- ✅ No warnings

### 3. Environment Setup

```bash
# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x+cu118
CUDA: True
```

## Quick Start Commands

### Test Installation
```bash
# Run tests
pytest tests/test_transnnmil_v2.py -v

# Should show: 16/16 passed
```

### Start Training (Recommended)
```bash
# 2-branch baseline (fastest, good performance)
python scripts/train_v2_0.py \
  --data_dir panda/features_resnet50_300patches \
  --splits_file panda/splits.json \
  --num_classes 5 \
  --epochs 50 \
  --batch_size 1 \
  --lr 1e-4 \
  --branches transmil hierarchical \
  --output_dir results/v2_0_2branch_baseline
```

### Monitor Training
```bash
# Terminal 1: Training
python scripts/train_v2_0.py ...

# Terminal 2: TensorBoard
tensorboard --logdir results/v2_0_2branch_baseline/logs
```

Open browser: http://localhost:6006

## Expected Timeline

1. **Transfer data**: 10-30 min (depends on method)
2. **Setup environment**: 5-10 min
3. **Verify setup**: 2 min
4. **Training (2-branch)**: 2-3 hours (GPU)
5. **Evaluation**: 5 min

Total: ~3-4 hours

## Troubleshooting

### Data Transfer Issues
```bash
# Check file count
ls panda/features_resnet50_300patches/*.h5 | wc -l
# Should be: 1365

# Check file integrity (sample)
python -c "import h5py; f = h5py.File('panda/features_resnet50_300patches/0005f7aaab2800f6170c399693a96917.h5', 'r'); print(list(f.keys()))"
# Should show: ['coords', 'features', 'label', 'slide_id']
```

### CUDA Issues
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt

# Check specific imports
python -c "from models.transnnmil_v2 import TransnnMILv2; print('OK')"
```

## What's Ready

✅ **Code:**
- TransnnMIL v2.0 implementation (3-branch)
- Training script with all options
- Evaluation script
- Visualization tools
- 116/116 tests passing

✅ **Data:**
- 1,365 PANDA slides
- Features extracted (ResNet50)
- Coordinates normalized
- Labels added
- Splits created

✅ **Documentation:**
- Architecture guide
- Training guide
- API reference
- Model card
- This setup guide

## What to Do on Other PC

1. Clone repo
2. Transfer data files
3. Setup environment
4. Verify setup
5. Start training
6. Monitor results
7. Report back with metrics

## Contact/Questions

If issues:
1. Check `TRAINING_SETUP.md` for detailed commands
2. Check `docs/TRANSNNMIL_V2_TRAINING.md` for training guide
3. Run verification script
4. Check logs in `results/<experiment>/logs/`

Good luck! 🚀
