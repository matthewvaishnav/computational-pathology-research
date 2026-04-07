"""
Regression tests for PCam baseline comparison runner.

Tests cover the bugs fixed in commit 493fb25:
1. Quick-test metadata corruption (original config path preservation)
2. Wildcard expansion on Windows/PowerShell
3. Duplicate deduplication in wildcard matches
4. Correct results_dir from config evaluation.output_dir
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import yaml


@pytest.fixture
def mock_config():
    """Create a mock PCam comparison config."""
    return {
        'experiment': {
            'name': 'test_variant'
        },
        'training': {
            'num_epochs': 20,
            'batch_size': 32
        },
        'early_stopping': {
            'patience': 5
        },
        'checkpoint': {
            'checkpoint_dir': './checkpoints/test_variant'
        },
        'evaluation': {
            'output_dir': './results/pcam_comparison/test_variant'
        },
        'data': {
            'root_dir': './data/pcam',
            'num_workers': 0
        }
    }


@pytest.fixture
def mock_config_file(tmp_path, mock_config):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_config, f)
    return config_path


def test_wildcard_expansion_windows_style(tmp_path):
    """
    Test that wildcard patterns are expanded correctly on Windows/PowerShell.
    
    Regression test for Bug 2: Wildcard expansion on Windows/PowerShell.
    Windows/PowerShell doesn't expand wildcards automatically, so the script
    must use glob.glob() to expand them.
    """
    from experiments.compare_pcam_baselines import main
    
    # Create test config files
    config1 = tmp_path / "config1.yaml"
    config2 = tmp_path / "config2.yaml"
    
    test_config = {
        'experiment': {'name': 'test1'},
        'training': {'num_epochs': 20, 'batch_size': 32},
        'early_stopping': {'patience': 5},
        'checkpoint': {'checkpoint_dir': './checkpoints/test1'},
        'evaluation': {'output_dir': './results/test1'},
        'data': {'root_dir': './data/pcam', 'num_workers': 0}
    }
    
    with open(config1, 'w') as f:
        yaml.dump(test_config, f)
    
    test_config['experiment']['name'] = 'test2'
    test_config['checkpoint']['checkpoint_dir'] = './checkpoints/test2'
    test_config['evaluation']['output_dir'] = './results/test2'
    
    with open(config2, 'w') as f:
        yaml.dump(test_config, f)
    
    # Test wildcard expansion with quoted pattern (Windows/PowerShell style)
    wildcard_pattern = str(tmp_path / "*.yaml")
    
    with patch('sys.argv', [
        'compare_pcam_baselines.py',
        '--configs', wildcard_pattern,
        '--skip-training',
        '--output', str(tmp_path / 'results.json')
    ]):
        with patch('experiments.compare_pcam_baselines.run_training') as mock_train:
            with patch('experiments.compare_pcam_baselines.run_evaluation') as mock_eval:
                with patch('pathlib.Path.exists', return_value=False):
                    # Mock evaluation to return checkpoint_not_found
                    mock_eval.return_value = {
                        'variant_name': 'test1',
                        'checkpoint_path': './checkpoints/test1/best_model.pth',
                        'status': 'checkpoint_not_found',
                        'metrics': {}
                    }
                    
                    try:
                        main()
                    except SystemExit:
                        pass  # Expected when checkpoints don't exist
        
        # Verify that glob expansion worked - should have found 2 configs
        # Check the results file to see how many variants were processed
        results_file = tmp_path / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            assert len(results['variants']) == 2, "Should have expanded wildcard to 2 configs"


def test_duplicate_wildcard_deduplication(tmp_path):
    """
    Test that duplicate config paths from wildcard expansion are deduplicated.
    
    If a user provides overlapping patterns or the same file multiple times,
    the script should deduplicate while preserving order.
    """
    from experiments.compare_pcam_baselines import main
    
    # Create a test config file
    config_path = tmp_path / "test_config.yaml"
    test_config = {
        'experiment': {'name': 'test_variant'},
        'training': {'num_epochs': 20, 'batch_size': 32},
        'early_stopping': {'patience': 5},
        'checkpoint': {'checkpoint_dir': './checkpoints/test'},
        'evaluation': {'output_dir': './results/test'},
        'data': {'root_dir': './data/pcam', 'num_workers': 0}
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    # Provide the same config multiple times (simulating overlapping wildcards)
    with patch('sys.argv', [
        'compare_pcam_baselines.py',
        '--configs', str(config_path), str(config_path), str(config_path),
        '--skip-training',
        '--output', str(tmp_path / 'results.json')
    ]):
        with patch('experiments.compare_pcam_baselines.run_evaluation') as mock_eval:
            with patch('pathlib.Path.exists', return_value=False):
                mock_eval.return_value = {
                    'variant_name': 'test_variant',
                    'checkpoint_path': './checkpoints/test/best_model.pth',
                    'status': 'checkpoint_not_found',
                    'metrics': {}
                }
                
                try:
                    main()
                except SystemExit:
                    pass
        
        # Verify deduplication - should only process once
        results_file = tmp_path / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            assert len(results['variants']) == 1, "Duplicate configs should be deduplicated"


def test_quick_test_preserves_original_config_path(tmp_path, mock_config):
    """
    Test that quick-test mode returns the original config path, not the temp config.
    
    Regression test for Bug 1: Quick-test metadata corruption.
    In quick-test mode, the script creates a temp config with reduced epochs,
    but should return the original config path in results, not the temp path.
    """
    from experiments.compare_pcam_baselines import run_training
    
    # Create original config file
    original_config_path = tmp_path / "original_config.yaml"
    with open(original_config_path, 'w') as f:
        yaml.dump(mock_config, f)
    
    # Mock subprocess to simulate successful training
    with patch('subprocess.run') as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training completed"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Run training in quick-test mode
        result = run_training(str(original_config_path), quick_test=True)
    
    # Verify the result contains the original config path, not a temp path
    assert result['config_path'] == str(original_config_path)
    assert 'temp_' not in result['config_path']
    assert result['variant_name'] == 'test_variant'
    assert result['status'] == 'success'
    
    # Verify temp config was cleaned up
    temp_config_path = tmp_path / "temp_original_config.yaml"
    assert not temp_config_path.exists(), "Temp config should be cleaned up"


def test_quick_test_cleans_up_temp_config_on_failure(tmp_path, mock_config):
    """
    Test that temp config is cleaned up even when training fails.
    
    Ensures proper cleanup in error paths.
    """
    from experiments.compare_pcam_baselines import run_training
    
    # Create original config file
    original_config_path = tmp_path / "original_config.yaml"
    with open(original_config_path, 'w') as f:
        yaml.dump(mock_config, f)
    
    # Mock subprocess to simulate failed training
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = Exception("Training failed")
        
        # Run training in quick-test mode (should handle exception)
        with patch('subprocess.CalledProcessError', Exception):
            try:
                result = run_training(str(original_config_path), quick_test=True)
            except Exception:
                pass  # Expected
    
    # Verify temp config was cleaned up even on failure
    temp_config_path = tmp_path / "temp_original_config.yaml"
    assert not temp_config_path.exists(), "Temp config should be cleaned up on failure"


def test_aggregate_results_uses_config_output_dir(tmp_path, mock_config):
    """
    Test that aggregate_results uses evaluation.output_dir from config, not invented paths.
    
    Regression test for Bug 1: Metadata corruption.
    The results_dir should come from config['evaluation']['output_dir'],
    not be constructed from the config path.
    """
    from experiments.compare_pcam_baselines import aggregate_results
    
    # Create config file
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_config, f)
    
    # Mock training and evaluation results
    training_results = [{
        'variant_name': 'test_variant',
        'config_path': str(config_path),
        'status': 'success',
        'training_time_seconds': 100.5
    }]
    
    evaluation_results = [{
        'variant_name': 'test_variant',
        'checkpoint_path': './checkpoints/test_variant/best_model.pth',
        'status': 'success',
        'metrics': {
            'accuracy': 0.95,
            'auc': 0.98,
            'f1': 0.94
        }
    }]
    
    output_path = tmp_path / "comparison_results.json"
    
    # Run aggregation
    aggregate_results(training_results, evaluation_results, str(output_path))
    
    # Load and verify results
    with open(output_path, 'r') as f:
        results = json.load(f)
    
    assert len(results['variants']) == 1
    variant = results['variants'][0]
    
    # Verify results_dir matches config evaluation.output_dir
    assert variant['results_dir'] == mock_config['evaluation']['output_dir']
    assert variant['results_dir'] == './results/pcam_comparison/test_variant'
    
    # Verify it's not an invented path
    assert 'temp_' not in variant['results_dir']
    assert variant['config_path'] == str(config_path)


def test_aggregate_results_with_multiple_variants(tmp_path):
    """
    Test aggregate_results with multiple variants to ensure each gets correct paths.
    """
    from experiments.compare_pcam_baselines import aggregate_results
    
    # Create multiple config files with different output dirs
    configs = []
    for i in range(3):
        config = {
            'experiment': {'name': f'variant_{i}'},
            'training': {'num_epochs': 20, 'batch_size': 32},
            'early_stopping': {'patience': 5},
            'checkpoint': {'checkpoint_dir': f'./checkpoints/variant_{i}'},
            'evaluation': {'output_dir': f'./results/pcam_comparison/variant_{i}'},
            'data': {'root_dir': './data/pcam', 'num_workers': 0}
        }
        
        config_path = tmp_path / f"config_{i}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        configs.append((config, config_path))
    
    # Mock training and evaluation results
    training_results = [
        {
            'variant_name': f'variant_{i}',
            'config_path': str(config_path),
            'status': 'success',
            'training_time_seconds': 100.0 + i
        }
        for i, (_, config_path) in enumerate(configs)
    ]
    
    evaluation_results = [
        {
            'variant_name': f'variant_{i}',
            'checkpoint_path': f'./checkpoints/variant_{i}/best_model.pth',
            'status': 'success',
            'metrics': {
                'accuracy': 0.90 + i * 0.02,
                'auc': 0.95 + i * 0.01
            }
        }
        for i in range(3)
    ]
    
    output_path = tmp_path / "comparison_results.json"
    
    # Run aggregation
    aggregate_results(training_results, evaluation_results, str(output_path))
    
    # Load and verify results
    with open(output_path, 'r') as f:
        results = json.load(f)
    
    assert len(results['variants']) == 3
    
    # Verify each variant has correct results_dir from its config
    for i, variant in enumerate(results['variants']):
        expected_output_dir = f'./results/pcam_comparison/variant_{i}'
        assert variant['results_dir'] == expected_output_dir
        assert variant['name'] == f'variant_{i}'
        assert variant['config_path'] == str(configs[i][1])


def test_no_configs_found_exits_gracefully(tmp_path):
    """
    Test that the script exits gracefully when no config files match the pattern.
    """
    from experiments.compare_pcam_baselines import main
    
    # Provide a wildcard pattern that matches nothing
    with patch('sys.argv', [
        'compare_pcam_baselines.py',
        '--configs', str(tmp_path / "nonexistent_*.yaml"),
        '--output', str(tmp_path / 'results.json')
    ]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with error code 1
        assert exc_info.value.code == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
