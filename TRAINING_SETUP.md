# TransnnMIL v2.0 Training Setup Guide

## Quick Start (Other PC)

### 1. Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Required Files

**Code:**
- `models/transnnmil_v2.py` - Main model
- `models/adaptive_pruning.py` - Adaptive pruning module
- `models/hierarchical_pooling.py` - Hierarchical pooling
- `models/topology_branch.py` - Topology branch
- `scripts/train_v2_0.py` - Training script
- `utils/` - Utility modules

**Data:**
- `panda/features_resnet50_300patches/*.h5` - 1,365 feature files (normalized, labeled)
- `panda/splits.json` - Train/val/test splits
- `panda/train.csv` - Original labels (reference)

### 3. Training Commands

#### Option 1: Baseline 2-Branch (Recommended First)
```bash
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

**Expected:**
- Training time: ~2-3 hours (GPU)
- Memory: ~6GB VRAM
- Parameters: 4.9M

#### Option 2: Full 3-Branch
```bash
python scripts/train_v2_0.py \
  --data_dir panda/features_resnet50_300patches \
  --splits_file panda/splits.json \
  --num_classes 5 \
  --epochs 50 \
  --batch_size 1 \
  --lr 1e-4 \
  --branches transmil hierarchical topology \
  --output_dir results/v2_0_3branch_full
```

**Expected:**
- Training time: ~3-4 hours (GPU)
- Memory: ~8GB VRAM
- Parameters: 6.8M
- Note: Adaptive pruning disabled in 3-branch (needs batched implementation)

#### Option 3: TransMIL Only (Fast Baseline)
```bash
python scripts/train_v2_0.py \
  --data_dir panda/features_resnet50_300patches \
  --splits_file panda/splits.json \
  --num_classes 5 \
  --epochs 50 \
  --batch_size 1 \
  --lr 1e-4 \
  --branches transmil \
  --output_dir results/v2_0_transmil_only
```

**Expected:**
- Training time: ~1-2 hours (GPU)
- Memory: ~4GB VRAM
- Parameters: 2.1M

### 4. Monitor Training

Training outputs:
- `results/<experiment>/checkpoints/` - Model checkpoints
- `results/<experiment>/logs/` - TensorBoard logs
- `results/<experiment>/metrics.json` - Training metrics

View logs:
```bash
tensorboard --logdir results/<experiment>/logs
```

### 5. Evaluation

After training:
```bash
python scripts/evaluate_v2_0.py \
  --checkpoint results/<experiment>/checkpoints/best_model.pth \
  --data_dir panda/features_resnet50_300patches \
  --splits_file panda/splits.json \
  --split test
```

### 6. Troubleshooting

**CUDA out of memory:**
- Reduce `--batch_size` to 1 (already minimum)
- Use `--gradient_accumulation_steps 4` for effective batch size 4
- Disable topology branch

**Slow training:**
- Check GPU utilization: `nvidia-smi`
- Reduce `--num_workers` if CPU bottleneck
- Use mixed precision: `--amp`

**NaN loss:**
- Reduce learning rate: `--lr 5e-5`
- Add gradient clipping: `--grad_clip 1.0`
- Check data normalization

## Dataset Info

**PANDA (Prostate cANcer graDe Assessment)**
- 1,365 slides (from 10,616 total)
- Features: ResNet50 (2048-D)
- Patches: ~60 per slide (18-131 range)
- Labels: ISUP grades 0-4 (5 classes)
- Splits: 955 train / 204 val / 206 test (70/15/15)

**Class Distribution (Full Dataset):**
- Grade 0: ~40%
- Grade 1: ~10%
- Grade 2: ~15%
- Grade 3: ~15%
- Grade 4: ~10%
- Grade 5: ~10%

Note: Imbalanced - consider class weights or focal loss.

## Architecture Details

**2-Branch (TransMIL + Hierarchical):**
- TransMIL: Self-attention over patches
- Hierarchical: Multi-scale region pooling (4 levels)
- Fusion: Concatenate → FC → Softmax
- Params: 4.9M

**3-Branch (+ Topology):**
- Topology: k-NN graph (k=5) + GCN (2 layers)
- Captures spatial relationships
- Fusion: Concatenate all 3 → FC → Softmax
- Params: 6.8M

**Adaptive Pruning:**
- Prunes low-attention patches during training
- Reduces computation ~30%
- Currently disabled in 3-branch (needs batched implementation)
- Works in 2-branch and TransMIL-only

## Next Steps

1. **Run baseline** (2-branch, 50 epochs)
2. **Analyze results** (confusion matrix, per-class metrics)
3. **Tune hyperparameters** (LR, batch size, class weights)
4. **Try 3-branch** if baseline works
5. **Implement batched pruning** for 3-branch optimization

## References

- Architecture: `docs/TRANSNNMIL_V2_ARCHITECTURE.md`
- Training guide: `docs/TRANSNNMIL_V2_TRAINING.md`
- API reference: `docs/TRANSNNMIL_V2_API.md`
- Model card: `docs/MODEL_CARD_V2.md`
