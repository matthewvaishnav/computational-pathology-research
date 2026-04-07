"""
Tests for model profiler to verify checkpoint loading compatibility.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from scripts.model_profiler import ModelProfiler
from src.models.multimodal import MultimodalFusionModel


def test_profiler_loads_state_dict_only_checkpoint():
    """
    Test that ModelProfiler can load state-dict-only checkpoints from training.
    
    This is a regression test for the issue where profiler fallback defaults
    didn't match MultimodalFusionModel() defaults, causing state dict size mismatches.
    """
    # Create a model with default parameters (as training would)
    model = MultimodalFusionModel()
    
    # Save a checkpoint with only model_state_dict (as SupervisedTrainer does)
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 10,
            'best_val_metric': 0.85,
        }
        torch.save(checkpoint, checkpoint_path)
    
    try:
        # Load with profiler - this should work without errors
        profiler = ModelProfiler(
            checkpoint_path=checkpoint_path,
            batch_size=2,
            verbose=False,
        )
        
        # Verify model loaded successfully
        assert profiler.model is not None
        assert isinstance(profiler.model, MultimodalFusionModel)
        
        # Verify model can do forward pass
        batch = profiler.create_dummy_inputs()
        with torch.no_grad():
            output = profiler.model(batch)
        
        assert output.shape == (2, 256)  # batch_size=2, embed_dim=256
        
    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


def test_profiler_loads_checkpoint_with_config():
    """
    Test that ModelProfiler can load checkpoints that include config.
    """
    # Create a model with custom config
    custom_embed_dim = 128
    model = MultimodalFusionModel(embed_dim=custom_embed_dim)
    
    # Save checkpoint with config
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'embed_dim': custom_embed_dim,
                'dropout': 0.1,
            },
            'epoch': 5,
        }
        torch.save(checkpoint, checkpoint_path)
    
    try:
        # Load with profiler
        profiler = ModelProfiler(
            checkpoint_path=checkpoint_path,
            batch_size=4,
            embed_dim=custom_embed_dim,
            verbose=False,
        )
        
        # Verify model loaded
        assert profiler.model is not None
        
        # Verify forward pass
        batch = profiler.create_dummy_inputs()
        with torch.no_grad():
            output = profiler.model(batch)
        
        assert output.shape == (4, custom_embed_dim)
        
    finally:
        Path(checkpoint_path).unlink()


def test_profiler_loads_checkpoint_with_nested_hydra_config():
    """
    Test that ModelProfiler can load checkpoints with Hydra-style nested config.
    """
    custom_embed_dim = 128
    model = MultimodalFusionModel(embed_dim=custom_embed_dim, dropout=0.1)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model': {
                    'embed_dim': custom_embed_dim,
                    'dropout': 0.1,
                }
            },
            'epoch': 5,
        }
        torch.save(checkpoint, checkpoint_path)

    try:
        profiler = ModelProfiler(
            checkpoint_path=checkpoint_path,
            batch_size=2,
            verbose=False,
        )

        batch = profiler.create_dummy_inputs()
        with torch.no_grad():
            output = profiler.model(batch)

        assert output.shape == (2, custom_embed_dim)
    finally:
        Path(checkpoint_path).unlink()


def test_profiler_state_dict_compatibility():
    """
    Test that profiler-loaded model has compatible state dict with original.
    """
    # Create original model
    original_model = MultimodalFusionModel(embed_dim=256)
    original_state = original_model.state_dict()
    
    # Save state dict only
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        torch.save({'model_state_dict': original_state}, checkpoint_path)
    
    try:
        # Load with profiler
        profiler = ModelProfiler(
            checkpoint_path=checkpoint_path,
            embed_dim=256,
            verbose=False,
        )
        
        loaded_state = profiler.model.state_dict()
        
        # Verify all keys match
        assert set(original_state.keys()) == set(loaded_state.keys())
        
        # Verify all shapes match
        for key in original_state.keys():
            assert original_state[key].shape == loaded_state[key].shape, \
                f"Shape mismatch for {key}: {original_state[key].shape} vs {loaded_state[key].shape}"
        
    finally:
        Path(checkpoint_path).unlink()


def test_profiler_with_different_embed_dims():
    """
    Test that profiler works with different embedding dimensions.
    """
    for embed_dim in [128, 256, 512]:
        model = MultimodalFusionModel(embed_dim=embed_dim)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        
        try:
            profiler = ModelProfiler(
                checkpoint_path=checkpoint_path,
                batch_size=2,
                embed_dim=embed_dim,
                verbose=False,
            )
            
            batch = profiler.create_dummy_inputs()
            with torch.no_grad():
                output = profiler.model(batch)
            
            assert output.shape == (2, embed_dim)
            
        finally:
            Path(checkpoint_path).unlink()


def test_profiler_fallback_uses_model_defaults():
    """
    Test that profiler fallback path uses MultimodalFusionModel defaults exactly.
    
    This ensures that when no config is in checkpoint, the profiler creates
    a model with the same architecture as MultimodalFusionModel() would create.
    """
    # Create model with defaults
    default_model = MultimodalFusionModel()
    default_state = default_model.state_dict()
    
    # Save without config
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        torch.save({'model_state_dict': default_state}, checkpoint_path)
    
    try:
        # Load with profiler (no config in checkpoint)
        profiler = ModelProfiler(
            checkpoint_path=checkpoint_path,
            verbose=False,
        )
        
        profiler_state = profiler.model.state_dict()
        
        # All keys should match
        assert set(default_state.keys()) == set(profiler_state.keys()), \
            "State dict keys don't match - profiler fallback doesn't match model defaults"
        
        # All shapes should match
        for key in default_state.keys():
            default_shape = default_state[key].shape
            profiler_shape = profiler_state[key].shape
            assert default_shape == profiler_shape, \
                f"Shape mismatch for {key}: default={default_shape}, profiler={profiler_shape}"
        
    finally:
        Path(checkpoint_path).unlink()


def test_profiler_execution_time_profiling():
    """
    Test that execution time profiling works.
    """
    model = MultimodalFusionModel()
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    
    try:
        profiler = ModelProfiler(
            checkpoint_path=checkpoint_path,
            batch_size=2,
            verbose=False,
        )
        
        # Run execution time profiling
        stats = profiler.profile_execution_time(num_iterations=5)
        
        # Verify stats returned
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        
        # Verify reasonable values
        assert stats['mean'] > 0
        assert stats['min'] > 0
        
    finally:
        Path(checkpoint_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
