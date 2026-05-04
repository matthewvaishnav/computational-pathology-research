"""
Self-Supervised Pre-Training System for Foundation Models
Implements SimCLR, MoCo, and DINO for histopathology data
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class PreTrainingConfig:
    """Configuration for self-supervised pre-training"""

    method: str = "simclr"  # simclr, moco, dino
    temperature: float = 0.07
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # MoCo specific
    moco_momentum: float = 0.999
    queue_size: int = 65536

    # DINO specific
    dino_momentum: float = 0.996
    dino_warmup_teacher_temp: float = 0.04
    dino_teacher_temp: float = 0.04
    dino_warmup_epochs: int = 10

    # Training
    warmup_epochs: int = 10
    save_freq: int = 10
    log_freq: int = 100

    # Distributed
    world_size: int = 1
    rank: int = 0
    distributed: bool = False


@dataclass
class AugmentationConfig:
    """Histopathology-specific augmentation configuration"""

    # Color augmentation (critical for histopathology)
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # Geometric augmentation
    rotation_degrees: int = 90
    flip_prob: float = 0.5

    # Stain normalization
    stain_normalize: bool = True
    target_stain: str = "he"  # H&E staining

    # Gaussian blur
    blur_prob: float = 0.1
    blur_sigma: Tuple[float, float] = (0.1, 2.0)


class HistopathologyAugmentation:
    """Histopathology-specific data augmentation with advanced techniques"""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation and return two views"""
        view1 = self._augment_single(image)
        view2 = self._augment_single(image)
        return view1, view2

    def _augment_single(self, image: torch.Tensor) -> torch.Tensor:
        """Apply single augmentation with histopathology-specific enhancements"""
        # Stain normalization (critical for histopathology)
        if self.config.stain_normalize and torch.rand(1) < 0.7:
            image = self._stain_normalize(image)
        
        # Color jitter (adapted for H&E staining)
        if torch.rand(1) < 0.8:
            image = self._histopathology_color_jitter(image)

        # Random rotation (90, 180, 270 degrees - preserves tissue structure)
        if torch.rand(1) < 0.5:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])

        # Random flip (horizontal only - vertical flip less common in pathology)
        if torch.rand(1) < self.config.flip_prob:
            image = torch.flip(image, dims=[-1])

        # Gaussian blur (simulate focus variations)
        if torch.rand(1) < self.config.blur_prob:
            image = self._gaussian_blur(image)
            
        # Elastic deformation (simulate tissue deformation)
        if torch.rand(1) < 0.3:
            image = self._elastic_deformation(image)
            
        # Random erasing (simulate artifacts/bubbles)
        if torch.rand(1) < 0.2:
            image = self._random_erasing(image)

        return image

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jitter augmentation"""
        # Brightness
        brightness_factor = (
            1
            + torch.rand(1) * 2 * self.config.color_jitter_brightness
            - self.config.color_jitter_brightness
        )
        image = image * brightness_factor

        # Contrast
        contrast_factor = (
            1
            + torch.rand(1) * 2 * self.config.color_jitter_contrast
            - self.config.color_jitter_contrast
        )
        mean = image.mean(dim=[-2, -1], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        # Saturation (convert to HSV, modify S, convert back)
        # Simplified saturation adjustment
        saturation_factor = (
            1
            + torch.rand(1) * 2 * self.config.color_jitter_saturation
            - self.config.color_jitter_saturation
        )
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        image = image * saturation_factor + gray.unsqueeze(0) * (1 - saturation_factor)

        return torch.clamp(image, 0, 1)

    def _stain_normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Apply stain normalization for H&E consistency"""
        # Simplified stain normalization using color constancy
        # In practice, would use Macenko or Reinhard normalization
        
        # Convert to LAB color space approximation
        image_np = image.permute(1, 2, 0).numpy()
        
        # Apply simple color constancy (Gray World assumption)
        mean_rgb = image_np.mean(axis=(0, 1))
        target_mean = np.array([0.7, 0.5, 0.7])  # Target H&E appearance
        
        # Scale factors
        scale_factors = target_mean / (mean_rgb + 1e-8)
        scale_factors = np.clip(scale_factors, 0.5, 2.0)  # Limit scaling
        
        # Apply scaling
        normalized = image_np * scale_factors
        normalized = np.clip(normalized, 0, 1)
        
        return torch.from_numpy(normalized).permute(2, 0, 1).float()
    
    def _histopathology_color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jitter adapted for histopathology"""
        # More conservative color jitter for medical images
        
        # Brightness (less aggressive than natural images)
        brightness_factor = 1 + torch.rand(1) * 0.2 - 0.1  # ±10%
        image = image * brightness_factor

        # Contrast (preserve diagnostic features)
        contrast_factor = 1 + torch.rand(1) * 0.3 - 0.15  # ±15%
        mean = image.mean(dim=[-2, -1], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        # Saturation (important for H&E differentiation)
        saturation_factor = 1 + torch.rand(1) * 0.4 - 0.2  # ±20%
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        image = image * saturation_factor + gray.unsqueeze(0) * (1 - saturation_factor)

        # Hue shift (small shifts to simulate staining variations)
        hue_shift = torch.rand(1) * 0.1 - 0.05  # ±5% hue shift
        image = self._apply_hue_shift(image, hue_shift)

        return torch.clamp(image, 0, 1)
    
    def _apply_hue_shift(self, image: torch.Tensor, hue_shift: float) -> torch.Tensor:
        """Apply small hue shift to simulate staining variations"""
        # Simplified hue shift in RGB space
        # In practice, would convert to HSV, shift hue, convert back
        
        # Create rotation matrix for hue shift
        cos_h = torch.cos(hue_shift * 2 * np.pi)
        sin_h = torch.sin(hue_shift * 2 * np.pi)
        
        # Apply hue rotation (simplified)
        r, g, b = image[0], image[1], image[2]
        
        new_r = r * cos_h - g * sin_h
        new_g = r * sin_h + g * cos_h
        new_b = b  # Blue channel less affected in H&E
        
        return torch.stack([new_r, new_g, new_b])
    
    def _elastic_deformation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation to simulate tissue deformation"""
        # Simplified elastic deformation using random displacement fields
        h, w = image.shape[-2:]
        
        # Create random displacement field
        displacement_x = torch.randn(h // 8, w // 8) * 2
        displacement_y = torch.randn(h // 8, w // 8) * 2
        
        # Upsample displacement field
        displacement_x = F.interpolate(
            displacement_x.unsqueeze(0).unsqueeze(0), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        displacement_y = F.interpolate(
            displacement_y.unsqueeze(0).unsqueeze(0), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        # Create sampling grid
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, w),
            torch.linspace(-1, 1, h),
            indexing='xy'
        )
        
        # Apply displacement
        grid_x = grid_x + displacement_x / w * 0.1  # Small displacement
        grid_y = grid_y + displacement_y / h * 0.1
        
        # Stack grid
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Apply grid sampling
        deformed = F.grid_sample(
            image.unsqueeze(0), 
            grid, 
            mode='bilinear', 
            padding_mode='reflection',
            align_corners=False
        ).squeeze(0)
        
        return deformed
    
    def _random_erasing(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random erasing to simulate artifacts"""
        if torch.rand(1) < 0.5:
            return image
            
        h, w = image.shape[-2:]
        
        # Random erasing parameters
        area_ratio = torch.rand(1) * 0.02 + 0.01  # 1-3% of image
        aspect_ratio = torch.rand(1) * 0.5 + 0.5   # 0.5-1.0
        
        # Calculate dimensions
        area = h * w * area_ratio
        erase_h = int(torch.sqrt(area / aspect_ratio))
        erase_w = int(torch.sqrt(area * aspect_ratio))
        
        # Random position
        top = torch.randint(0, max(1, h - erase_h), (1,)).item()
        left = torch.randint(0, max(1, w - erase_w), (1,)).item()
        
        # Random fill value (simulate bubble/artifact)
        fill_value = torch.rand(3, 1, 1) * 0.3 + 0.7  # Light colored artifacts
        
        # Apply erasing
        image[:, top:top+erase_h, left:left+erase_w] = fill_value
        
        return image

    def _gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur"""
        # Simplified Gaussian blur implementation
        kernel_size = 3
        sigma = (
            torch.rand(1) * (self.config.blur_sigma[1] - self.config.blur_sigma[0])
            + self.config.blur_sigma[0]
        )

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable convolution
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

        # Pad and convolve
        padding = kernel_size // 2
        image_padded = F.pad(
            image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect"
        )
        blurred = F.conv2d(image_padded, kernel_2d, groups=3)

        return blurred.squeeze(0)


class SimCLRLoss(nn.Module):
    """SimCLR contrastive loss"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss

        Args:
            features: [2*batch_size, feature_dim] - concatenated positive pairs

        Returns:
            Contrastive loss
        """
        batch_size = features.shape[0] // 2

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(
            features.device
        )

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float("inf"))

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class MoCoQueue:
    """Memory queue for MoCo"""

    def __init__(self, feature_dim: int, queue_size: int):
        self.queue_size = queue_size
        self.queue = torch.randn(feature_dim, queue_size)
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0

    def dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys"""
        batch_size = keys.shape[0]

        if not (self.queue_size % batch_size == 0):
            raise ValueError("Queue size must be divisible by batch size")

        # Replace oldest keys
        self.queue[:, self.queue_ptr : self.queue_ptr + batch_size] = keys.T
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size


class SelfSupervisedPreTrainer:
    """Self-supervised pre-training system"""

    def __init__(
        self,
        model: nn.Module,
        config: PreTrainingConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
    ):
        self.model = model
        self.config = config
        self.augmentation = HistopathologyAugmentation(augmentation_config or AugmentationConfig())

        # Initialize method-specific components
        self._init_method_components()

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )

        # Logging
        self.logger = logging.getLogger(__name__)
        self.metrics = defaultdict(list)

    def _init_method_components(self):
        """Initialize method-specific components"""
        if self.config.method == "simclr":
            self.criterion = SimCLRLoss(self.config.temperature)

        elif self.config.method == "moco":
            # Create momentum encoder
            self.momentum_encoder = self._create_momentum_encoder()
            self.queue = MoCoQueue(self.model.config.feature_dim, self.config.queue_size)
            self.criterion = nn.CrossEntropyLoss()

        elif self.config.method == "dino":
            # Create teacher network
            self.teacher = self._create_momentum_encoder()
            self.teacher_temp_schedule = self._create_teacher_temp_schedule()

        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _create_momentum_encoder(self) -> nn.Module:
        """Create momentum encoder for MoCo/DINO"""
        momentum_encoder = type(self.model)(self.model.config)
        momentum_encoder.load_state_dict(self.model.state_dict())

        # Freeze momentum encoder
        for param in momentum_encoder.parameters():
            param.requires_grad = False

        return momentum_encoder

    def _create_teacher_temp_schedule(self) -> List[float]:
        """Create teacher temperature schedule for DINO"""
        warmup_epochs = self.config.dino_warmup_epochs
        total_epochs = self.config.num_epochs

        schedule = []
        for epoch in range(total_epochs):
            if epoch < warmup_epochs:
                # Linear warmup
                temp = (
                    self.config.dino_warmup_teacher_temp
                    + (self.config.dino_teacher_temp - self.config.dino_warmup_teacher_temp)
                    * epoch
                    / warmup_epochs
                )
            else:
                temp = self.config.dino_teacher_temp
            schedule.append(temp)

        return schedule

    def pretrain(
        self,
        dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute self-supervised pre-training with distributed support"""
        num_epochs = num_epochs or self.config.num_epochs

        # Initialize distributed training if configured
        if self.config.distributed and not dist.is_initialized():
            self._init_distributed_training()

        # Wrap model for distributed training
        if self.config.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=True
            )

        # Create distributed sampler if needed
        sampler = None
        if self.config.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )

        # Create data loader with distributed sampler
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.logger.info(f"Starting {self.config.method} pre-training for {num_epochs} epochs")
        if self.config.distributed:
            self.logger.info(f"Distributed training: rank {self.config.rank}/{self.config.world_size}")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Set epoch for distributed sampler
            if sampler is not None:
                sampler.set_epoch(epoch)

            # Training
            train_metrics = self._train_epoch(dataloader, epoch)

            # Validation (only on rank 0 to avoid duplication)
            if validation_dataset is not None and (not self.config.distributed or self.config.rank == 0):
                val_metrics = self._validate_epoch(validation_dataset, epoch)
                train_metrics.update(val_metrics)

            # Update scheduler
            self.scheduler.step()

            # Logging (only on rank 0)
            if not self.config.distributed or self.config.rank == 0:
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Loss: {train_metrics['loss']:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save metrics
                for key, value in train_metrics.items():
                    self.metrics[key].append(value)

                # Save checkpoint
                if (epoch + 1) % self.config.save_freq == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch)

            # Synchronize processes
            if self.config.distributed:
                dist.barrier()

        total_time = time.time() - start_time
        
        if not self.config.distributed or self.config.rank == 0:
            self.logger.info(f"Pre-training completed in {total_time:.2f}s")

        return {
            "metrics": dict(self.metrics),
            "total_time": total_time,
            "final_loss": self.metrics["loss"][-1] if self.metrics["loss"] else None,
        }

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train single epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Get images (assuming batch is just images for unlabeled data)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.cuda() if torch.cuda.is_available() else images

            # Apply augmentation
            view1_list, view2_list = [], []
            for img in images:
                view1, view2 = self.augmentation(img)
                view1_list.append(view1)
                view2_list.append(view2)

            view1 = torch.stack(view1_list)
            view2 = torch.stack(view2_list)

            # Forward pass based on method
            if self.config.method == "simclr":
                loss = self._simclr_forward(view1, view2)
            elif self.config.method == "moco":
                loss = self._moco_forward(view1, view2)
            elif self.config.method == "dino":
                loss = self._dino_forward(view1, view2, epoch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update momentum encoder for MoCo/DINO
            if self.config.method in ["moco", "dino"]:
                self._update_momentum_encoder(epoch)

            total_loss += loss.item()
            num_batches += 1

            # Logging
            if batch_idx % self.config.log_freq == 0:
                self.logger.debug(
                    f"Epoch {epoch+1} Batch {batch_idx}/{len(dataloader)} - "
                    f"Loss: {loss.item():.4f}"
                )

        return {"loss": total_loss / num_batches}

    def _simclr_forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """SimCLR forward pass"""
        # Extract features
        features1 = self.model.extract_features(view1.unsqueeze(1)).squeeze(
            1
        )  # Remove patch dimension
        features2 = self.model.extract_features(view2.unsqueeze(1)).squeeze(1)

        # Concatenate positive pairs
        features = torch.cat([features1, features2], dim=0)

        # Compute loss
        loss = self.criterion(features)

        return loss

    def _moco_forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """MoCo forward pass"""
        # Query features (from main encoder)
        q = self.model.extract_features(view1.unsqueeze(1)).squeeze(1)
        q = F.normalize(q, dim=1)

        # Key features (from momentum encoder)
        with torch.no_grad():
            k = self.momentum_encoder.extract_features(view2.unsqueeze(1)).squeeze(1)
            k = F.normalize(k, dim=1)

        # Positive logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.queue.clone().detach()])

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.config.temperature

        # Labels (positive pairs are at index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Update queue
        self.queue.dequeue_and_enqueue(k)

        loss = self.criterion(logits, labels)

        return loss

    def _dino_forward(self, view1: torch.Tensor, view2: torch.Tensor, epoch: int) -> torch.Tensor:
        """DINO forward pass"""
        # Student features
        student_out1 = self.model.extract_features(view1.unsqueeze(1)).squeeze(1)
        student_out2 = self.model.extract_features(view2.unsqueeze(1)).squeeze(1)

        # Teacher features
        with torch.no_grad():
            teacher_out1 = self.teacher.extract_features(view1.unsqueeze(1)).squeeze(1)
            teacher_out2 = self.teacher.extract_features(view2.unsqueeze(1)).squeeze(1)

        # Temperature for teacher
        teacher_temp = self.teacher_temp_schedule[epoch]

        # Compute DINO loss
        loss = 0
        for s_out, t_out in [(student_out1, teacher_out2), (student_out2, teacher_out1)]:
            # Student probabilities (with temperature 0.1)
            s_prob = F.softmax(s_out / 0.1, dim=1)

            # Teacher probabilities (with scheduled temperature)
            t_prob = F.softmax(t_out / teacher_temp, dim=1)

            # Cross-entropy loss
            loss += -torch.sum(t_prob * torch.log(s_prob + 1e-8), dim=1).mean()

        return loss / 2

    def _init_distributed_training(self):
        """Initialize distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.config.rank = int(os.environ['RANK'])
            self.config.world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Set device for current process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.rank % torch.cuda.device_count())
            self.model = self.model.cuda()
            if hasattr(self, 'momentum_encoder'):
                self.momentum_encoder = self.momentum_encoder.cuda()
        
        self.logger.info(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")

    def _update_momentum_encoder(self, epoch: int)::
        """Update momentum encoder parameters"""
        if self.config.method == "moco":
            momentum = self.config.moco_momentum
        elif self.config.method == "dino":
            momentum = self.config.dino_momentum
        else:
            return

        # Update momentum encoder
        for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)

    def _validate_epoch(self, validation_dataset: Dataset, epoch: int) -> Dict[str, float]:
        """Validate single epoch using linear probing"""
        # Simplified validation - just compute features and return dummy metrics
        self.model.eval()

        val_loader = DataLoader(
            validation_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.cuda() if torch.cuda.is_available() else images
                features = self.model.extract_features(images.unsqueeze(1)).squeeze(1)
                total_samples += images.shape[0]

        return {"val_samples": total_samples}

    def save_checkpoint(self, path: str, epoch: int):
        """Save comprehensive training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if not self.config.distributed 
                               else self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": dict(self.metrics),
            "random_state": torch.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

        if hasattr(self, "momentum_encoder"):
            checkpoint["momentum_encoder_state_dict"] = (
                self.momentum_encoder.state_dict() if not self.config.distributed
                else self.momentum_encoder.module.state_dict()
            )

        if hasattr(self, "queue"):
            checkpoint["queue_state"] = {
                "queue": self.queue.queue.clone(),
                "queue_ptr": self.queue.queue_ptr
            }

        # Save with atomic write to prevent corruption
        temp_path = path + ".tmp"
        torch.save(checkpoint, temp_path)
        
        # Atomic rename
        if os.path.exists(temp_path):
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
            
        self.logger.info(f"Checkpoint saved: {path}")

    def resume_from_checkpoint(self, path: str) -> int:
        """Resume training from checkpoint with full state restoration"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Load model state
        if self.config.distributed:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load momentum encoder if exists
        if hasattr(self, "momentum_encoder") and "momentum_encoder_state_dict" in checkpoint:
            if self.config.distributed:
                self.momentum_encoder.module.load_state_dict(checkpoint["momentum_encoder_state_dict"])
            else:
                self.momentum_encoder.load_state_dict(checkpoint["momentum_encoder_state_dict"])

        # Load queue state for MoCo
        if hasattr(self, "queue") and "queue_state" in checkpoint:
            self.queue.queue = checkpoint["queue_state"]["queue"]
            self.queue.queue_ptr = checkpoint["queue_state"]["queue_ptr"]

        # Restore random states for reproducibility
        if "random_state" in checkpoint:
            torch.set_rng_state(checkpoint["random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])

        # Load metrics
        self.metrics = defaultdict(list, checkpoint["metrics"])

        epoch = checkpoint["epoch"]
        self.logger.info(f"Resumed from checkpoint: {path}, epoch: {epoch}")

        return epoch

    def auto_resume_latest_checkpoint(self, checkpoint_dir: str = "checkpoints") -> Optional[int]:
        """Automatically resume from the latest checkpoint if available"""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
            
        # Find all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoint_files:
            return None
            
        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split("_")[-1])
            except:
                return 0
                
        latest_checkpoint = max(checkpoint_files, key=extract_epoch)
        
        try:
            epoch = self.resume_from_checkpoint(str(latest_checkpoint))
            self.logger.info(f"Auto-resumed from latest checkpoint: {latest_checkpoint}")
            return epoch
        except Exception as e:
            self.logger.error(f"Failed to resume from {latest_checkpoint}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from multi_disease_model import create_foundation_model

    # Create model
    model = create_foundation_model()

    # Create pre-trainer
    config = PreTrainingConfig(method="simclr", batch_size=32, num_epochs=10, learning_rate=1e-3)

    pretrainer = SelfSupervisedPreTrainer(model, config)

    # Mock dataset
    class MockDataset(Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            return torch.randn(3, 224, 224)

    dataset = MockDataset()

    # Start pre-training
    results = pretrainer.pretrain(dataset)
    print(f"Pre-training completed. Final loss: {results['final_loss']}")
