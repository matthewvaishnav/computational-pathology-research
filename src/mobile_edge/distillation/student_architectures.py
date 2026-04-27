"""
Student Architecture Designs for Knowledge Distillation

Lightweight student models for mobile/edge deployment via distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging


@dataclass
class StudentConfig:
    """Student model config"""
    architecture: str = 'mobilenet_v3_small'  # mobilenet_v3_small/tiny/efficientnet_b0/squeezenet
    width_multiplier: float = 1.0             # Channel width scale
    depth_multiplier: float = 1.0             # Depth scale
    input_size: int = 224                     # Input resolution
    num_classes: int = 2                      # Output classes
    dropout: float = 0.2                      # Dropout rate
    use_se: bool = True                       # Squeeze-excitation blocks
    activation: str = 'relu'                  # relu/swish/hardswish
    
    def __post_init__(self):
        assert 0 < self.width_multiplier <= 1.0
        assert 0 < self.depth_multiplier <= 1.0


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small for edge deployment"""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        # Inverted residual settings: [kernel, exp, out, SE, NL, stride]
        settings = [
            [3, 16, 16, True, 'RE', 2],
            [3, 72, 24, False, 'RE', 2],
            [3, 88, 24, False, 'RE', 1],
            [5, 96, 40, True, 'HS', 2],
            [5, 240, 40, True, 'HS', 1],
            [5, 240, 40, True, 'HS', 1],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 48, True, 'HS', 1],
            [5, 288, 96, True, 'HS', 2],
            [5, 576, 96, True, 'HS', 1],
            [5, 576, 96, True, 'HS', 1],
        ]
        
        # Scale channels
        wm = config.width_multiplier
        
        # First conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(16 * wm), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * wm)),
            nn.Hardswish()
        )
        
        # Inverted residual blocks
        self.blocks = nn.ModuleList()
        in_ch = int(16 * wm)
        
        for k, exp, out, se, nl, s in settings:
            out_ch = int(out * wm)
            exp_ch = int(exp * wm)
            self.blocks.append(
                InvertedResidual(in_ch, out_ch, exp_ch, k, s, se, nl)
            )
            in_ch = out_ch
        
        # Final layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, int(576 * wm), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(576 * wm)),
            nn.Hardswish()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(int(576 * wm), int(1024 * wm)),
            nn.Hardswish(),
            nn.Dropout(config.dropout),
            nn.Linear(int(1024 * wm), config.num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetV3Tiny(nn.Module):
    """Ultra-lightweight MobileNetV3 for extreme edge"""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        # Minimal settings
        settings = [
            [3, 16, 16, False, 'RE', 2],
            [3, 48, 24, False, 'RE', 2],
            [5, 72, 40, True, 'HS', 2],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 96, True, 'HS', 2],
        ]
        
        wm = config.width_multiplier
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(16 * wm), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(16 * wm)),
            nn.Hardswish()
        )
        
        self.blocks = nn.ModuleList()
        in_ch = int(16 * wm)
        
        for k, exp, out, se, nl, s in settings:
            out_ch = int(out * wm)
            exp_ch = int(exp * wm)
            self.blocks.append(
                InvertedResidual(in_ch, out_ch, exp_ch, k, s, se, nl)
            )
            in_ch = out_ch
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, int(288 * wm), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(288 * wm)),
            nn.Hardswish()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(int(288 * wm), int(512 * wm)),
            nn.Hardswish(),
            nn.Dropout(config.dropout),
            nn.Linear(int(512 * wm), config.num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EfficientNetB0Student(nn.Module):
    """EfficientNet-B0 student (scaled down)"""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        # MBConv settings: [kernel, exp_ratio, out, layers, stride]
        settings = [
            [3, 1, 16, 1, 1],
            [3, 6, 24, 2, 2],
            [5, 6, 40, 2, 2],
            [3, 6, 80, 3, 2],
            [5, 6, 112, 3, 1],
            [5, 6, 192, 4, 2],
            [3, 6, 320, 1, 1],
        ]
        
        wm = config.width_multiplier
        dm = config.depth_multiplier
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * wm), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * wm)),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList()
        in_ch = int(32 * wm)
        
        for k, exp_r, out, layers, s in settings:
            out_ch = int(out * wm)
            num_layers = max(1, int(layers * dm))
            
            for i in range(num_layers):
                stride = s if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(in_ch, out_ch, k, stride, exp_r, config.use_se)
                )
                in_ch = out_ch
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, int(1280 * wm), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(1280 * wm)),
            nn.SiLU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(int(1280 * wm), config.num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SqueezeNetStudent(nn.Module):
    """SqueezeNet student (very lightweight)"""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            
            FireModule(64, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            
            FireModule(128, 32, 128, 128),
            FireModule(256, 32, 128, 128),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            FireModule(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Conv2d(512, config.num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV3)"""
    
    def __init__(self, in_ch, out_ch, exp_ch, kernel, stride, use_se, activation):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        
        act = nn.Hardswish() if activation == 'HS' else nn.ReLU(inplace=True)
        
        layers = []
        
        # Expand
        if exp_ch != in_ch:
            layers.extend([
                nn.Conv2d(in_ch, exp_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(exp_ch),
                act
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(exp_ch, exp_ch, kernel, stride, kernel//2, groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            act
        ])
        
        # SE
        if use_se:
            layers.append(SEBlock(exp_ch))
        
        # Project
        layers.extend([
            nn.Conv2d(exp_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck conv (EfficientNet)"""
    
    def __init__(self, in_ch, out_ch, kernel, stride, exp_ratio, use_se):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        exp_ch = in_ch * exp_ratio
        
        layers = []
        
        # Expand
        if exp_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch),
                nn.SiLU()
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(exp_ch, exp_ch, kernel, stride, kernel//2, groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            nn.SiLU()
        ])
        
        # SE
        if use_se:
            layers.append(SEBlock(exp_ch))
        
        # Project
        layers.extend([
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FireModule(nn.Module):
    """Fire module (SqueezeNet)"""
    
    def __init__(self, in_ch, squeeze_ch, expand1x1_ch, expand3x3_ch):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, squeeze_ch, 1),
            nn.ReLU(inplace=True)
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand1x1_ch, 1),
            nn.ReLU(inplace=True)
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_ch, expand3x3_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)


def create_student_model(config: StudentConfig) -> nn.Module:
    """Factory for student models"""
    if config.architecture == 'mobilenet_v3_small':
        return MobileNetV3Small(config)
    elif config.architecture == 'mobilenet_v3_tiny':
        return MobileNetV3Tiny(config)
    elif config.architecture == 'efficientnet_b0':
        return EfficientNetB0Student(config)
    elif config.architecture == 'squeezenet':
        return SqueezeNetStudent(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable params"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Model size MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def compare_student_architectures(num_classes: int = 2) -> Dict[str, Dict]:
    """Compare student arch stats"""
    results = {}
    
    architectures = [
        ('mobilenet_v3_small', 1.0),
        ('mobilenet_v3_small', 0.75),
        ('mobilenet_v3_tiny', 1.0),
        ('efficientnet_b0', 0.75),
        ('squeezenet', 1.0),
    ]
    
    for arch, width in architectures:
        config = StudentConfig(
            architecture=arch,
            width_multiplier=width,
            num_classes=num_classes
        )
        
        model = create_student_model(config)
        
        name = f"{arch}_w{width}"
        results[name] = {
            'params': count_parameters(model),
            'size_mb': get_model_size_mb(model),
            'architecture': arch,
            'width_multiplier': width
        }
    
    return results


# Medical AI optimized students
def create_medical_student(size: str = 'small', num_classes: int = 2) -> nn.Module:
    """Medical AI student models"""
    
    if size == 'tiny':
        config = StudentConfig(
            architecture='mobilenet_v3_tiny',
            width_multiplier=0.75,
            num_classes=num_classes,
            dropout=0.3,  # Higher dropout for medical
            use_se=True
        )
    elif size == 'small':
        config = StudentConfig(
            architecture='mobilenet_v3_small',
            width_multiplier=0.75,
            num_classes=num_classes,
            dropout=0.3,
            use_se=True
        )
    elif size == 'medium':
        config = StudentConfig(
            architecture='efficientnet_b0',
            width_multiplier=0.75,
            depth_multiplier=0.8,
            num_classes=num_classes,
            dropout=0.3,
            use_se=True
        )
    else:
        raise ValueError(f"Unknown size: {size}")
    
    return create_student_model(config)
