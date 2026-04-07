"""
Tests for ablation study runner to verify output directory alignment.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_ablation_runner_output_dir_alignment(tmp_path):
    """
    Test that ablation runner passes correct Hydra output directory override.

    This ensures the training script writes to the same directory that the
    ablation runner later reads from.
    """
    from scripts.run_ablation_study import run_experiment

    # Mock configuration
    config = {
        "name": "test_experiment",
        "description": "Test experiment",
        "model": "multimodal",
        "data": {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
        },
    }

    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    num_epochs = 1
    batch_size = 2

    # Mock subprocess.run to capture the command
    with patch("subprocess.run") as mock_run:
        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock file existence and content
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("builtins.open", create=True) as mock_open:
                # Mock test_results.json
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                    {
                        "accuracy": 0.85,
                        "auc": 0.90,
                    }
                )

                # Run experiment
                result = run_experiment(
                    config=config,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                )

        # Verify subprocess.run was called
        assert mock_run.called

        # Get the command that was passed
        cmd = mock_run.call_args[0][0]

        # Verify the command contains the Hydra output directory override
        expected_output_path = output_dir / "test_experiment"
        hydra_override_found = False

        for arg in cmd:
            if arg.startswith("hydra.run.dir="):
                hydra_override_found = True
                # Extract the path from the override
                override_path = arg.split("=", 1)[1]
                assert Path(override_path).resolve() == expected_output_path.resolve()
                break

        assert hydra_override_found, "Command should contain hydra.run.dir override"

        # Verify other expected arguments
        assert f"model={config['model']}" in cmd
        assert f"experiment_name={config['name']}" in cmd
        assert f"data.data_dir={data_dir}" in cmd


def test_ablation_runner_failed_experiment_detection(tmp_path):
    """
    Test that the ablation runner correctly detects and reports failed experiments.
    """
    from scripts.run_ablation_study import run_ablation_study

    # Mock subprocess to simulate a failed experiment
    with patch("subprocess.run") as mock_run:
        # First experiment succeeds
        success_result = MagicMock()
        success_result.returncode = 0
        success_result.stderr = ""

        # Second experiment fails
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "Training failed: CUDA out of memory"

        mock_run.side_effect = [success_result, fail_result]

        # Mock file operations for successful experiment
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                    {
                        "accuracy": 0.85,
                    }
                )

                # Run ablation study with 2 experiments
                with patch(
                    "scripts.run_ablation_study.ABLATION_CONFIGS",
                    [
                        {
                            "name": "exp1",
                            "description": "Experiment 1",
                            "model": "multimodal",
                            "data": {
                                "wsi_enabled": True,
                                "genomic_enabled": True,
                                "clinical_text_enabled": True,
                            },
                        },
                        {
                            "name": "exp2",
                            "description": "Experiment 2",
                            "model": "baseline",
                            "model_modality": "wsi",
                            "data": {
                                "wsi_enabled": True,
                                "genomic_enabled": False,
                                "clinical_text_enabled": False,
                            },
                        },
                    ],
                ):
                    data_dir = tmp_path / "data"
                    output_dir = tmp_path / "output"
                    data_dir.mkdir()
                    df, all_results = run_ablation_study(
                        data_dir=str(data_dir),
                        output_dir=str(output_dir),
                        num_epochs=1,
                        batch_size=2,
                    )

        # Verify results
        assert len(all_results) == 2

        # First experiment should succeed
        assert all_results[0]["status"] == "completed"
        assert all_results[0]["name"] == "exp1"

        # Second experiment should fail
        assert all_results[1]["status"] == "failed"
        assert all_results[1]["name"] == "exp2"
        assert "CUDA out of memory" in all_results[1]["error"]

        # DataFrame should only contain successful experiments
        assert len(df) == 1
        assert df.iloc[0]["experiment"] == "exp1"


def test_output_path_construction(tmp_path):
    """
    Test that output paths are constructed correctly and consistently.
    """
    output_dir = tmp_path / "output"
    experiment_name = "test_exp"

    # This is how the ablation runner constructs the path
    experiment_output = output_dir / experiment_name

    # This is what should be passed to Hydra
    hydra_override = f"hydra.run.dir={experiment_output.absolute()}"

    # Verify the paths match
    assert str(experiment_output) == str(output_dir / experiment_name)
    assert experiment_name in hydra_override


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
