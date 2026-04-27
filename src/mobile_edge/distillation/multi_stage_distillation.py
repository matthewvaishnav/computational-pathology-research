"""
Multi-Stage Knowledge Distillation

Progressive distillation through intermediate teacher models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import copy

from .distillation_loss import DistillationLoss, DistillationConfig
from .temperature_scheduling import TemperatureScheduler, TemperatureScheduleConfig


@dataclass
class MultiStageConfig:
    """Multi-stage distillation config"""
    num_stages: int = 3                   # Number of distillation stages
    stage_epochs: List[int] = None        # Epochs per stage
    intermediate_sizes: List[float] = None # Size ratios for intermediate models
    temperature_schedule: str = 'cosine'   # Temperature schedule type
    initial_temp: float = 8.0             # Initial temperature
    final_temp: float = 2.0               # Final temperature
    alpha: float = 0.7                    # Distillation loss weight
    learning_rate: float = 1e-4           # Learning rate
    
    def __post_init__(self):
        if self.stage_epochs is None:
            self.stage_epochs = [30, 30, 40]  # Default 3 stages
        if self.intermediate_sizes is None:
            self.intermediate_sizes = [0.75, 0.5, 0.25]  # Progressive compression


class MultiStageDistiller:
    """
    Multi-stage knowledge distillation
    
    Process:
    1. Teacher → Intermediate1 (75% size)
    2. Intermediate1 → Intermediate2 (50% size)
    3. Intermediate2 → Student (25% size)
    
    Benefits:
    - Smoother knowledge transfer
    - Better final accuracy
    - Handles large capacity gaps
    """
    
    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Stage models
        self.teacher = None
        self.intermediates = []
        self.student = None
        
        # Training history
        self.stage_history = []
    
    def distill(self, teacher: nn.Module,
               student: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device,
               create_intermediate_fn: callable) -> nn.Module:
        """
        Perform multi-stage distillation
        
        Args:
            teacher: Teacher model
            student: Final student model
            train_loader: Training data
            val_loader: Validation data
            device: Compute device
            create_intermediate_fn: Function to create intermediate models
            
        Returns:
            Trained student model
        """
        
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        
        # Create intermediate models
        self.intermediates = []
        for size_ratio in self.config.intermediate_sizes[:-1]:
            intermediate = create_intermediate_fn(size_ratio)
            self.intermediates.append(intermediate.to(device))
        
        # Add final student as last intermediate
        self.intermediates.append(self.student)
        
        self.logger.info(f"Multi-stage distillation: {self.config.num_stages} stages")
        
        # Stage 1: Teacher → Intermediate1
        current_teacher = self.teacher
        
        for stage_idx, (intermediate, epochs) in enumerate(
            zip(self.intermediates, self.config.stage_epochs)
        ):
            self.logger.info(f"Stage {stage_idx + 1}/{self.config.num_stages}")
            
            # Train this stage
            trained_model, stage_stats = self._train_stage(
                teacher=current_teacher,
                student=intermediate,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                stage_idx=stage_idx
            )
            
            # Record history
            self.stage_history.append({
                'stage': stage_idx + 1,
                'epochs': epochs,
                'stats': stage_stats
            })
            
            # Next stage uses this as teacher
            current_teacher = trained_model
        
        self.logger.info("Multi-stage distillation complete")
        
        return self.student
    
    def _train_stage(self, teacher: nn.Module,
                    student: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    device: torch.device,
                    epochs: int,
                    stage_idx: int) -> Tuple[nn.Module, Dict]:
        """Train single distillation stage"""
        
        # Setup
        teacher.eval()
        student.train()
        
        # Distillation loss
        dist_config = DistillationConfig(
            temperature=self.config.initial_temp,
            alpha=self.config.alpha,
            beta=1.0 - self.config.alpha,
            loss_type='kl'
        )
        criterion = DistillationLoss(dist_config)
        
        # Optimizer
        optimizer = optim.Adam(student.parameters(), lr=self.config.learning_rate)
        
        # Temperature scheduler
        temp_config = TemperatureScheduleConfig(
            initial_temp=self.config.initial_temp,
            final_temp=self.config.final_temp,
            schedule_type=self.config.temperature_schedule,
            total_epochs=epochs,
            warmup_epochs=5
        )
        temp_scheduler = TemperatureScheduler(temp_config)
        
        # Training loop
        best_acc = 0.0
        stats = {
            'train_loss': [],
            'val_acc': [],
            'temperatures': []
        }
        
        for epoch in range(epochs):
            # Update temperature
            temp = temp_scheduler.step(epoch)
            criterion.config.temperature = temp
            stats['temperatures'].append(temp)
            
            # Train epoch
            train_loss = self._train_epoch(
                teacher, student, train_loader, device, criterion, optimizer
            )
            stats['train_loss'].append(train_loss)
            
            # Validate
            val_acc = self._validate(student, val_loader, device)
            stats['val_acc'].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Stage {stage_idx + 1} Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Temp: {temp:.2f}"
                )
        
        stats['best_acc'] = best_acc
        
        return student, stats
    
    def _train_epoch(self, teacher: nn.Module,
                    student: nn.Module,
                    train_loader: DataLoader,
                    device: torch.device,
                    criterion: DistillationLoss,
                    optimizer: optim.Optimizer) -> float:
        """Train single epoch"""
        
        student.train()
        teacher.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            student_logits = student(data)
            
            # Loss
            loss, _ = criterion(student_logits, teacher_logits, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self, model: nn.Module,
                 val_loader: DataLoader,
                 device: torch.device) -> float:
        """Validate model"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def get_stage_history(self) -> List[Dict]:
        """Get training history for all stages"""
        return self.stage_history


class ProgressiveDistiller:
    """
    Progressive distillation with gradual capacity reduction
    
    Continuously reduces model size while maintaining performance
    """
    
    def __init__(self, num_steps: int = 5, 
                 compression_ratio: float = 0.5):
        self.num_steps = num_steps
        self.compression_ratio = compression_ratio
        self.logger = logging.getLogger(__name__)
    
    def distill(self, teacher: nn.Module,
               target_size: float,
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device,
               create_model_fn: callable,
               epochs_per_step: int = 20) -> nn.Module:
        """
        Progressive distillation to target size
        
        Args:
            teacher: Initial teacher
            target_size: Target size ratio (e.g., 0.25 for 25%)
            train_loader: Training data
            val_loader: Validation data
            device: Device
            create_model_fn: Function(size_ratio) -> model
            epochs_per_step: Training epochs per step
            
        Returns:
            Final compressed model
        """
        
        # Calculate size schedule
        current_size = 1.0
        size_schedule = []
        
        while current_size > target_size:
            next_size = max(current_size * self.compression_ratio, target_size)
            size_schedule.append(next_size)
            current_size = next_size
        
        self.logger.info(f"Progressive distillation: {len(size_schedule)} steps")
        self.logger.info(f"Size schedule: {size_schedule}")
        
        # Progressive distillation
        current_teacher = teacher.to(device)
        
        for step_idx, size_ratio in enumerate(size_schedule):
            self.logger.info(f"Step {step_idx + 1}/{len(size_schedule)} - "
                           f"Size ratio: {size_ratio:.3f}")
            
            # Create student for this step
            student = create_model_fn(size_ratio).to(device)
            
            # Distill
            config = MultiStageConfig(
                num_stages=1,
                stage_epochs=[epochs_per_step],
                alpha=0.7
            )
            distiller = MultiStageDistiller(config)
            
            # Single-stage distillation
            trained_student = distiller._train_stage(
                teacher=current_teacher,
                student=student,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs_per_step,
                stage_idx=step_idx
            )[0]
            
            # Next step uses this as teacher
            current_teacher = trained_student
        
        return current_teacher


def create_multistage_distiller(num_stages: int = 3,
                                total_epochs: int = 100) -> MultiStageDistiller:
    """Factory for multi-stage distiller"""
    
    # Distribute epochs across stages
    epochs_per_stage = total_epochs // num_stages
    stage_epochs = [epochs_per_stage] * num_stages
    
    # Adjust last stage to use remaining epochs
    stage_epochs[-1] += total_epochs - sum(stage_epochs)
    
    config = MultiStageConfig(
        num_stages=num_stages,
        stage_epochs=stage_epochs,
        temperature_schedule='cosine',
        initial_temp=8.0,
        final_temp=2.0,
        alpha=0.7
    )
    
    return MultiStageDistiller(config)


# Medical AI optimized multi-stage
def create_medical_multistage_distiller(total_epochs: int = 150) -> MultiStageDistiller:
    """Medical AI multi-stage distiller"""
    
    config = MultiStageConfig(
        num_stages=4,                     # More stages for medical precision
        stage_epochs=[40, 40, 40, 30],   # Longer training per stage
        intermediate_sizes=[0.8, 0.6, 0.4, 0.25],  # Gradual compression
        temperature_schedule='cosine',
        initial_temp=6.0,                 # Moderate initial temp
        final_temp=2.5,                   # Higher final temp for precision
        alpha=0.75,                       # Higher teacher weight
        learning_rate=5e-5                # Lower LR for stability
    )
    
    return MultiStageDistiller(config)
