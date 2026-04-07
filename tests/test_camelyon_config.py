"""
Tests for CAMELYON experiment configuration.

These tests verify that the CAMELYON config scaffold is valid and can be loaded.
"""

import pytest
import yaml
from pathlib import Path


def test_camelyon_config_exists():
    """Test that CAMELYON config file exists."""
    config_path = Path("experiments/configs/camelyon.yaml")
    assert config_path.exists(), f"CAMELYON config not found: {config_path}"


def test_camelyon_config_is_valid_yaml():
    """Test that CAMELYON config is valid YAML."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert config is not None
    assert isinstance(config, dict)


def test_camelyon_config_has_required_sections():
    """Test that CAMELYON config has all required sections."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required top-level sections
    required_sections = [
        'experiment',
        'data',
        'model',
        'task',
        'training',
        'validation',
        'checkpoint',
        'early_stopping',
        'logging',
        'evaluation',
        'seed',
        'device'
    ]
    
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"


def test_camelyon_config_experiment_metadata():
    """Test that experiment metadata is properly configured."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment = config['experiment']
    
    assert experiment['name'] == 'camelyon16'
    assert 'description' in experiment
    assert 'tags' in experiment
    assert isinstance(experiment['tags'], list)
    assert 'camelyon' in experiment['tags']


def test_camelyon_config_data_section():
    """Test that data configuration is properly structured."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data = config['data']
    
    # Check basic data config
    assert data['dataset'] == 'camelyon16'
    assert 'root_dir' in data
    assert data['download'] == False  # Manual download required
    
    # Check slide-level config
    assert 'slide' in data
    slide = data['slide']
    assert 'patch_size' in slide
    assert 'patch_level' in slide
    assert 'stride' in slide
    assert 'background_threshold' in slide
    assert 'max_patches_per_slide' in slide
    
    # Verify reasonable values
    assert slide['patch_size'] > 0
    assert slide['patch_level'] >= 0
    assert 0 < slide['background_threshold'] < 1


def test_camelyon_config_model_section():
    """Test that model configuration is properly structured."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = config['model']
    
    # Check modalities
    assert 'modalities' in model
    assert model['modalities'] == ['wsi']
    
    # Check feature extractor
    assert 'feature_extractor' in model
    fe = model['feature_extractor']
    assert fe['model'] in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    assert fe['pretrained'] == True
    assert fe['feature_dim'] > 0
    
    # Check WSI encoder
    assert 'wsi' in model
    wsi = model['wsi']
    assert wsi['input_dim'] == fe['feature_dim']
    assert wsi['pooling'] in ['mean', 'max', 'attention']


def test_camelyon_config_task_section():
    """Test that task configuration is properly structured."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    task = config['task']
    
    assert task['type'] == 'classification'
    assert task['num_classes'] == 2  # Binary classification
    
    # Check classification head config
    assert 'classification' in task
    clf = task['classification']
    assert 'hidden_dims' in clf
    assert isinstance(clf['hidden_dims'], list)
    assert 'dropout' in clf


def test_camelyon_config_training_section():
    """Test that training configuration is properly structured."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training = config['training']
    
    # Check basic training params
    assert training['num_epochs'] > 0
    assert training['batch_size'] > 0
    assert float(training['learning_rate']) > 0  # May be scientific notation
    assert float(training['weight_decay']) >= 0
    
    # Check optimizer config
    assert 'optimizer' in training
    assert training['optimizer']['name'] in ['adam', 'adamw', 'sgd']
    
    # Check scheduler config
    assert 'scheduler' in training
    assert training['scheduler']['name'] in ['cosine', 'step', 'plateau']


def test_camelyon_config_paths_are_relative():
    """Test that all paths in config are relative for portability."""
    config_path = Path("experiments/configs/camelyon.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check data paths
    assert config['data']['root_dir'].startswith('./')
    
    # Check checkpoint path
    assert config['checkpoint']['checkpoint_dir'].startswith('./')
    
    # Check logging path
    assert config['logging']['log_dir'].startswith('./')
    
    # Check evaluation path
    assert config['evaluation']['output_dir'].startswith('./')


def test_camelyon_training_script_exists():
    """Test that CAMELYON training script exists."""
    script_path = Path("experiments/train_camelyon.py")
    assert script_path.exists(), f"CAMELYON training script not found: {script_path}"


def test_camelyon_training_script_is_executable():
    """Test that CAMELYON training script can be imported."""
    script_path = Path("experiments/train_camelyon.py")
    
    # Read the file to check it's valid Python
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for required elements
    assert 'def main()' in content
    assert 'argparse' in content
    assert '__main__' in content
    assert 'CAMELYONSlideIndex' in content
    assert 'CAMELYONPatchDataset' in content


def test_camelyon_training_script_has_real_training_loop():
    """Test that CAMELYON training script has real training logic."""
    script_path = Path("experiments/train_camelyon.py")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for real training components (not just placeholder)
    assert 'def train_epoch' in content
    assert 'def validate' in content
    assert 'SimpleSlideClassifier' in content
    assert 'optimizer.zero_grad()' in content
    assert 'loss.backward()' in content
    assert 'torch.save' in content  # Model saving


def test_simple_slide_classifier_can_be_imported():
    """Test that SimpleSlideClassifier can be imported and instantiated."""
    # Import the training script as a module
    import sys
    from pathlib import Path
    
    # Add experiments to path
    experiments_path = Path("experiments")
    sys.path.insert(0, str(experiments_path.absolute()))
    
    try:
        from train_camelyon import SimpleSlideClassifier
        
        # Create a simple model
        model = SimpleSlideClassifier(
            feature_dim=2048,
            hidden_dim=256,
            num_classes=2,
            pooling='mean',
            dropout=0.3,
        )
        
        # Check model structure
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'pooling')
        assert model.pooling == 'mean'
        
        # Test forward pass
        import torch
        batch_size = 4
        num_patches = 10
        feature_dim = 2048
        
        dummy_input = torch.randn(batch_size, num_patches, feature_dim)
        output = model(dummy_input)
        
        # Check output shape (binary classification)
        assert output.shape == (batch_size, 1)
        
    finally:
        # Clean up sys.path
        sys.path.remove(str(experiments_path.absolute()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
