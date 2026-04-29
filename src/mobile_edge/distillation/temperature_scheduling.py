"""
Temperature Scheduling for Knowledge Distillation

Dynamic temperature adjustment during distillation training.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging


@dataclass
class TemperatureScheduleConfig:
    """Temperature schedule config"""
    initial_temp: float = 8.0             # Starting temperature
    final_temp: float = 1.0               # Ending temperature
    schedule_type: str = 'cosine'         # cosine/linear/exponential/step/adaptive
    total_epochs: int = 100               # Total training epochs
    warmup_epochs: int = 5                # Warmup period
    min_temp: float = 1.0                 # Minimum temperature
    max_temp: float = 10.0                # Maximum temperature
    step_size: int = 20                   # Steps for step schedule
    gamma: float = 0.5                    # Decay factor for step/exp
    adaptive_threshold: float = 0.01      # Accuracy change threshold for adaptive


class TemperatureScheduler:
    """
    Temperature scheduler for distillation
    
    Schedules:
    - Cosine: Smooth decay (recommended)
    - Linear: Constant rate decay
    - Exponential: Fast initial decay
    - Step: Discrete drops
    - Adaptive: Based on validation performance
    
    High temp early → soft targets, gradual learning
    Low temp late → sharp targets, fine-tuning
    """
    
    def __init__(self, config: TemperatureScheduleConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.current_epoch = 0
        self.current_temp = config.initial_temp
        self.history = []
        
        # Adaptive state
        self.prev_accuracy = 0.0
        self.stagnant_epochs = 0
    
    def step(self, epoch: int, accuracy: float = None) -> float:
        """
        Update temperature for current epoch
        
        Args:
            epoch: Current epoch
            accuracy: Validation accuracy (for adaptive)
            
        Returns:
            Current temperature
        """
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            # Warmup: keep initial temp
            self.current_temp = self.config.initial_temp
        else:
            # Apply schedule
            if self.config.schedule_type == 'cosine':
                self.current_temp = self._cosine_schedule(epoch)
            elif self.config.schedule_type == 'linear':
                self.current_temp = self._linear_schedule(epoch)
            elif self.config.schedule_type == 'exponential':
                self.current_temp = self._exponential_schedule(epoch)
            elif self.config.schedule_type == 'step':
                self.current_temp = self._step_schedule(epoch)
            elif self.config.schedule_type == 'adaptive':
                self.current_temp = self._adaptive_schedule(epoch, accuracy)
            else:
                raise ValueError(f"Unknown schedule: {self.config.schedule_type}")
        
        # Clamp to bounds
        self.current_temp = max(self.config.min_temp, 
                               min(self.config.max_temp, self.current_temp))
        
        # Record history
        self.history.append({
            'epoch': epoch,
            'temperature': self.current_temp,
            'accuracy': accuracy
        })
        
        return self.current_temp
    
    def _cosine_schedule(self, epoch: int) -> float:
        """Cosine annealing schedule"""
        
        progress = (epoch - self.config.warmup_epochs) / \
                  (self.config.total_epochs - self.config.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        # Cosine decay
        temp = self.config.final_temp + \
               0.5 * (self.config.initial_temp - self.config.final_temp) * \
               (1 + math.cos(math.pi * progress))
        
        return temp
    
    def _linear_schedule(self, epoch: int) -> float:
        """Linear decay schedule"""
        
        progress = (epoch - self.config.warmup_epochs) / \
                  (self.config.total_epochs - self.config.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        # Linear interpolation
        temp = self.config.initial_temp + \
               progress * (self.config.final_temp - self.config.initial_temp)
        
        return temp
    
    def _exponential_schedule(self, epoch: int) -> float:
        """Exponential decay schedule"""
        
        progress = (epoch - self.config.warmup_epochs) / \
                  (self.config.total_epochs - self.config.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        # Exponential decay
        temp = self.config.initial_temp * (self.config.gamma ** progress)
        temp = max(temp, self.config.final_temp)
        
        return temp
    
    def _step_schedule(self, epoch: int) -> float:
        """Step decay schedule"""
        
        num_steps = (epoch - self.config.warmup_epochs) // self.config.step_size
        
        # Step decay
        temp = self.config.initial_temp * (self.config.gamma ** num_steps)
        temp = max(temp, self.config.final_temp)
        
        return temp
    
    def _adaptive_schedule(self, epoch: int, accuracy: float) -> float:
        """Adaptive schedule based on performance"""
        
        if accuracy is None:
            # Fallback to cosine
            return self._cosine_schedule(epoch)
        
        # Check if accuracy improving
        acc_change = accuracy - self.prev_accuracy
        
        if acc_change < self.config.adaptive_threshold:
            # Stagnant: decrease temp faster
            self.stagnant_epochs += 1
            decay_factor = 1.0 - (0.1 * self.stagnant_epochs)
        else:
            # Improving: maintain or slow decay
            self.stagnant_epochs = 0
            decay_factor = 0.98
        
        # Update temp
        temp = self.current_temp * decay_factor
        temp = max(temp, self.config.final_temp)
        
        self.prev_accuracy = accuracy
        
        return temp
    
    def get_temperature(self) -> float:
        """Get current temperature"""
        return self.current_temp
    
    def get_history(self) -> List[Dict]:
        """Get temperature history"""
        return self.history
    
    def reset(self):
        """Reset scheduler state"""
        self.current_epoch = 0
        self.current_temp = self.config.initial_temp
        self.history = []
        self.prev_accuracy = 0.0
        self.stagnant_epochs = 0


class MultiStageTemperatureScheduler:
    """
    Multi-stage temperature scheduling
    
    Different temps for different training stages:
    - Stage 1: High temp, learn soft targets
    - Stage 2: Medium temp, balance soft/hard
    - Stage 3: Low temp, focus on hard targets
    """
    
    def __init__(self, stages: List[Tuple[int, float]]):
        """
        Args:
            stages: List of (epoch_end, temperature) tuples
        """
        self.stages = sorted(stages, key=lambda x: x[0])
        self.logger = logging.getLogger(__name__)
        
        self.current_epoch = 0
        self.current_temp = stages[0][1]
        self.history = []
    
    def step(self, epoch: int) -> float:
        """Update temperature for epoch"""
        
        self.current_epoch = epoch
        
        # Find current stage
        for epoch_end, temp in self.stages:
            if epoch <= epoch_end:
                self.current_temp = temp
                break
        else:
            # Past all stages, use last temp
            self.current_temp = self.stages[-1][1]
        
        self.history.append({
            'epoch': epoch,
            'temperature': self.current_temp
        })
        
        return self.current_temp
    
    def get_temperature(self) -> float:
        return self.current_temp


class CurriculumTemperatureScheduler:
    """
    Curriculum-based temperature scheduling
    
    Adjusts temp based on sample difficulty:
    - Easy samples: Lower temp
    - Hard samples: Higher temp
    """
    
    def __init__(self, base_temp: float = 4.0, 
                 difficulty_range: Tuple[float, float] = (0.5, 2.0)):
        self.base_temp = base_temp
        self.min_mult, self.max_mult = difficulty_range
        self.logger = logging.getLogger(__name__)
    
    def get_temperature(self, difficulty: torch.Tensor) -> torch.Tensor:
        """
        Get per-sample temperatures
        
        Args:
            difficulty: Sample difficulty scores [0, 1]
            
        Returns:
            Per-sample temperatures
        """
        
        # Map difficulty to temp multiplier
        mult = self.min_mult + difficulty * (self.max_mult - self.min_mult)
        
        # Apply to base temp
        temps = self.base_temp * mult
        
        return temps


def create_temperature_scheduler(schedule_type: str = 'cosine',
                                initial_temp: float = 8.0,
                                final_temp: float = 1.0,
                                total_epochs: int = 100) -> TemperatureScheduler:
    """Factory for temperature schedulers"""
    
    config = TemperatureScheduleConfig(
        initial_temp=initial_temp,
        final_temp=final_temp,
        schedule_type=schedule_type,
        total_epochs=total_epochs,
        warmup_epochs=5
    )
    
    return TemperatureScheduler(config)


def create_multistage_scheduler(total_epochs: int = 100) -> MultiStageTemperatureScheduler:
    """Create 3-stage temperature scheduler"""
    
    stages = [
        (total_epochs // 3, 8.0),      # Stage 1: High temp
        (2 * total_epochs // 3, 4.0),  # Stage 2: Medium temp
        (total_epochs, 2.0)            # Stage 3: Low temp
    ]
    
    return MultiStageTemperatureScheduler(stages)


# Medical AI optimized scheduling
def create_medical_temperature_scheduler(total_epochs: int = 100) -> TemperatureScheduler:
    """Medical AI temperature scheduler"""
    
    config = TemperatureScheduleConfig(
        initial_temp=6.0,              # Moderate initial temp
        final_temp=2.0,                # Higher final temp for medical precision
        schedule_type='cosine',        # Smooth decay
        total_epochs=total_epochs,
        warmup_epochs=10,              # Longer warmup for stability
        min_temp=2.0,                  # Higher min for medical accuracy
        max_temp=8.0
    )
    
    return TemperatureScheduler(config)


def visualize_schedule(scheduler: TemperatureScheduler, 
                      total_epochs: int = 100) -> List[float]:
    """Generate temperature schedule for visualization"""
    
    temps = []
    for epoch in range(total_epochs):
        temp = scheduler.step(epoch)
        temps.append(temp)
    
    return temps
