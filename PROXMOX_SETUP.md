# Proxmox GPU Training Setup Guide

**Hardware**: R7 CPU, RTX 3070, 16GB RAM, 1.7TB storage  
**Goal**: Run Phikon benchmark and offload training workloads

---

## Step 1: Download Ubuntu ISO

**Option A: Ubuntu 22.04 LTS (Recommended)**
```bash
# On your main PC, download:
wget https://releases.ubuntu.com/22.04/ubuntu-22.04.5-live-server-amd64.iso

# Or use browser:
# https://ubuntu.com/download/server
# Click "Download Ubuntu Server 22.04.5 LTS"
```

**Option B: Ubuntu 24.04 LTS (Latest)**
```bash
wget https://releases.ubuntu.com/24.04/ubuntu-24.04-live-server-amd64.iso
```

**Upload to Proxmox**:
1. Open Proxmox web UI: `https://proxmox-ip:8006`
2. Click your node → `local (proxmox)` → `ISO Images`
3. Click `Upload` → Select downloaded ISO
4. Wait for upload to complete

---

## Step 2: Create VM in Proxmox

**Via Web UI**:
1. Click `Create VM` (top right)
2. **General**:
   - VM ID: 100
   - Name: `histocore-training`
3. **OS**:
   - ISO: Select uploaded Ubuntu ISO
   - Type: Linux
   - Version: 6.x - 2.6 Kernel
4. **System**:
   - Machine: q35
   - BIOS: OVMF (UEFI)
   - Add EFI Disk: Yes
   - SCSI Controller: VirtIO SCSI single
5. **Disks**:
   - Bus/Device: SCSI 0
   - Storage: local-lvm
   - Size: 100 GB
   - Cache: Write back
   - Discard: Yes
6. **CPU**:
   - Cores: 4
   - Type: host
7. **Memory**:
   - Memory: 8192 MB (8GB)
   - Minimum: 2048 MB
8. **Network**:
   - Bridge: vmbr0
   - Model: VirtIO
9. Click `Finish`

---

## Step 3: GPU Passthrough (RTX 3070)

**Enable IOMMU** (if not already done):
```bash
# SSH into Proxmox host
ssh root@proxmox-ip

# Edit GRUB
nano /etc/default/grub

# For AMD CPU (R7), change this line:
GRUB_CMDLINE_LINUX_DEFAULT="quiet amd_iommu=on iommu=pt"

# Update GRUB and reboot
update-grub
reboot
```

**Add GPU to VM**:
```bash
# SSH into Proxmox after reboot
ssh root@proxmox-ip

# Find GPU PCI address
lspci | grep -i nvidia
# Example output: 01:00.0 VGA compatible controller: NVIDIA RTX 3070

# Add GPU to VM (replace 01:00 with your address)
qm set 100 -hostpci0 01:00,pcie=1,rombar=0

# Start VM
qm start 100
```

---

## Step 4: Install Ubuntu in VM

1. Open VM console in Proxmox web UI
2. Select `Install Ubuntu Server`
3. **Installation options**:
   - Language: English
   - Keyboard: US
   - Network: DHCP (auto)
   - Storage: Use entire disk
   - Profile:
     - Name: `histocore`
     - Server name: `histocore-training`
     - Username: `histocore`
     - Password: (your choice)
   - SSH: Install OpenSSH server ✓
   - Featured snaps: None (skip)
4. Wait for installation (~5 minutes)
5. Reboot when prompted
6. Login with username/password

---

## Step 5: Install NVIDIA Drivers

```bash
# SSH into VM (get IP from Proxmox console)
ssh histocore@vm-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Reboot
sudo reboot

# After reboot, verify GPU
nvidia-smi
# Should show RTX 3070
```

---

## Step 6: Install PyTorch and HistoCore

```bash
# Install Python and dependencies
sudo apt install -y python3.10 python3-pip git wget curl

# Clone HistoCore
cd ~
git clone https://github.com/matthewvaishnav/computational-pathology-research.git
cd computational-pathology-research

# Install dependencies
pip install -r requirements.txt
pip install -e ".[foundation]"

# Verify PyTorch sees GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

---

## Step 7: Download PCam Dataset

```bash
cd ~/computational-pathology-research

# Download PCam (takes ~10 minutes, 2GB)
python scripts/download_pcam_manual.py --root-dir ./data/pcam_real

# Verify download
ls -lh data/pcam_real/
# Should show train/val/test directories
```

---

## Step 8: Run Phikon Benchmark

```bash
cd ~/computational-pathology-research

# Run training (30-45 minutes with caching)
python experiments/train_pcam.py --config configs/pcam_phikon.yaml

# Monitor progress
tail -f logs/pcam_phikon/training.log

# Or use tmux to run in background
sudo apt install -y tmux
tmux new -s training
python experiments/train_pcam.py --config configs/pcam_phikon.yaml
# Press Ctrl+B then D to detach
# Reattach with: tmux attach -t training
```

---

## Expected Output

**First Run** (~2 minutes):
```
Caching training features...
Caching train features: 100%|████████| 2048/2048 [01:45<00:00]
Caching validation features...
Caching val features: 100%|████████| 256/256 [00:13<00:00]
✓ Feature caching complete!
```

**Training** (~30-45 minutes):
```
Epoch 1/20: 100%|████████| 2048/2048 [02:15<00:00]
Train Loss: 0.3245, Train Acc: 87.3%
Val Loss: 0.2891, Val Acc: 89.1%, Val AUC: 0.9521
✓ New best val_auc: 0.9521

Epoch 20/20: 100%|████████| 2048/2048 [02:12<00:00]
Train Loss: 0.1234, Train Acc: 94.2%
Val Loss: 0.1567, Val Acc: 92.8%, Val AUC: 0.9789
✓ Training complete!
```

**Results Location**:
- Checkpoint: `checkpoints/pcam_phikon/best_model.pth`
- Logs: `logs/pcam_phikon/`
- Metrics: `results/pcam_phikon/metrics.json`

---

## Troubleshooting

**GPU not detected in VM**:
```bash
# Check if GPU is passed through
lspci | grep -i nvidia
# Should show RTX 3070

# If not, check Proxmox host:
ssh root@proxmox-ip
lspci | grep -i nvidia
qm config 100 | grep hostpci
```

**Out of memory**:
```bash
# Reduce batch size in config
nano configs/pcam_phikon.yaml
# Change: batch_size: 128 → batch_size: 64
```

**Slow training**:
```bash
# Check GPU utilization
nvidia-smi -l 1
# Should show ~90-100% GPU usage

# If low, increase num_workers
nano configs/pcam_phikon.yaml
# Change: num_workers: 4 → num_workers: 8
```

---

## Next Steps After Phikon Benchmark

1. **Compare foundation models**:
   ```bash
   # Train UNI (requires HuggingFace access)
   python experiments/train_pcam.py --config configs/pcam_uni.yaml
   
   # Train CONCH
   python experiments/train_pcam.py --config configs/pcam_conch.yaml
   ```

2. **Run on TCGA dataset**:
   ```bash
   # Download TCGA data
   python scripts/download_tcga.py
   
   # Train
   python experiments/train_tcga.py --config configs/tcga_phikon.yaml
   ```

3. **Set up continuous training**:
   ```bash
   # Install cron job to train nightly
   crontab -e
   # Add: 0 2 * * * cd ~/computational-pathology-research && python experiments/train_pcam.py --config configs/pcam_phikon.yaml
   ```

---

## Remote Access from Main PC

**SSH with key**:
```bash
# On main PC, copy SSH key to VM
ssh-copy-id histocore@vm-ip

# SSH without password
ssh histocore@vm-ip

# Copy results back to main PC
scp -r histocore@vm-ip:~/computational-pathology-research/results/ ./
```

**VS Code Remote**:
1. Install "Remote - SSH" extension
2. Connect to `histocore@vm-ip`
3. Open folder: `/home/histocore/computational-pathology-research`
4. Edit and run code remotely

---

## Resource Usage

**Expected**:
- GPU: 6-7GB VRAM (RTX 3070 has 8GB)
- RAM: 4-6GB (VM has 8GB)
- Disk: 20GB (dataset + checkpoints)
- Network: 2GB download (PCam dataset)

**Monitoring**:
```bash
# GPU usage
watch -n 1 nvidia-smi

# RAM/CPU usage
htop

# Disk usage
df -h
```

---

## Cost Savings

**Before**: Main GPU blocked for 2-3 hours per experiment  
**After**: Main GPU free, Proxmox box runs experiments 24/7

**Throughput**: 
- 1 experiment/day → 8 experiments/day (run overnight)
- Parallel experiments: Train Phikon + UNI + CONCH simultaneously

---

## Questions?

- Proxmox docs: https://pve.proxmox.com/wiki/Main_Page
- GPU passthrough: https://pve.proxmox.com/wiki/PCI_Passthrough
- HistoCore issues: https://github.com/matthewvaishnav/computational-pathology-research/issues
