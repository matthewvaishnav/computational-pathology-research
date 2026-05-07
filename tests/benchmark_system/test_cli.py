"""
Unit tests for Competitor Benchmark System CLI.

Tests command-line argument parsing, configuration loading,
and command execution logic.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.benchmark_system.cli import (
    create_config_from_args,
    create_parser,
    list_frameworks_command,
    load_config_from_file,
)
from experiments.benchmark_system.models import BenchmarkConfig, TaskSpecification


class TestCLIParser:
    """Test CLI argument parser."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.description is not None

    def test_parse_run_command_quick_mode(self):
        """Test parsing run command with quick mode."""
        parser = create_parser()
        args = parser.parse_args(["run", "--mode", "quick"])

        assert args.command == "run"
        assert args.mode == "quick"
        assert args.frameworks is None  # Default to all frameworks

    def test_parse_run_command_full_mode(self):
        """Test parsing run command with full mode."""
        parser = create_parser()
        args = parser.parse_args(["run", "--mode", "full"])

        assert args.command == "run"
        assert args.mode == "full"

    def test_parse_run_command_with_frameworks(self):
        """Test parsing run command with specific frameworks."""
        parser = create_parser()
        args = parser.parse_args(
            ["run", "--mode", "quick", "--frameworks", "HistoCore", "PathML"]
        )

        assert args.command == "run"
        assert args.frameworks == ["HistoCore", "PathML"]

    def test_parse_run_command_with_output_dir(self):
        """Test parsing run command with custom output directory."""
        parser = create_parser()
        args = parser.parse_args(["run", "--output-dir", "custom_results"])

        assert args.command == "run"
        assert args.output_dir == Path("custom_results")

    def test_parse_run_command_with_config(self):
        """Test parsing run command with config file."""
        parser = create_parser()
        args = parser.parse_args(["run", "--config", "config.yaml"])

        assert args.command == "run"
        assert args.config == Path("config.yaml")

    def test_parse_run_command_with_resume(self):
        """Test parsing run command with checkpoint resume."""
        parser = create_parser()
        args = parser.parse_args(["run", "--resume", "checkpoint.json"])

        assert args.command == "run"
        assert args.resume == Path("checkpoint.json")

    def test_parse_run_command_with_hyperparameters(self):
        """Test parsing run command with custom hyperparameters."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.001",
                "--num-epochs",
                "20",
            ]
        )

        assert args.command == "run"
        assert args.batch_size == 64
        assert args.learning_rate == 0.001
        assert args.num_epochs == 20

    def test_parse_validate_command(self):
        """Test parsing validate command."""
        parser = create_parser()
        args = parser.parse_args(["validate"])

        assert args.command == "validate"

    def test_parse_validate_command_with_output(self):
        """Test parsing validate command with output file."""
        parser = create_parser()
        args = parser.parse_args(["validate", "--output", "validation.json"])

        assert args.command == "validate"
        assert args.output == Path("validation.json")

    def test_parse_list_frameworks_command(self):
        """Test parsing list-frameworks command."""
        parser = create_parser()
        args = parser.parse_args(["list-frameworks"])

        assert args.command == "list-frameworks"

    def test_parse_resume_command(self):
        """Test parsing resume command."""
        parser = create_parser()
        args = parser.parse_args(["resume", "checkpoint.json"])

        assert args.command == "resume"
        assert args.checkpoint == Path("checkpoint.json")

    def test_parse_global_options(self):
        """Test parsing global options."""
        parser = create_parser()
        # Global options must come before the subcommand
        args = parser.parse_args(["--verbose", "--log-file", "benchmark.log", "run"])

        assert args.verbose is True
        assert args.log_file == Path("benchmark.log")


class TestConfigLoading:
    """Test configuration loading from files."""

    def test_load_config_from_json(self):
        """Test loading configuration from JSON file."""
        config_dict = {
            "mode": "quick",
            "frameworks": ["HistoCore", "PathML"],
            "quick_mode_epochs": 3,
            "quick_mode_samples": 1000,
            "output_dir": "results/test",
            "random_seed": 42,
            "task_spec": {
                "dataset_name": "PatchCamelyon",
                "data_root": "data/pcam",
                "model_architecture": "resnet18_transformer",
                "num_epochs": 10,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "random_seed": 42,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_dict, f)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path)

            assert isinstance(config, BenchmarkConfig)
            assert config.mode == "quick"
            assert config.frameworks == ["HistoCore", "PathML"]
            assert config.quick_mode_epochs == 3
            assert config.quick_mode_samples == 1000
            assert config.output_dir == Path("results/test")
            assert config.random_seed == 42

            assert isinstance(config.task_spec, TaskSpecification)
            assert config.task_spec.dataset_name == "PatchCamelyon"
            assert config.task_spec.data_root == Path("data/pcam")
            assert config.task_spec.model_architecture == "resnet18_transformer"

        finally:
            config_path.unlink()

    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        pytest.importorskip("yaml")

        import yaml

        config_dict = {
            "mode": "full",
            "frameworks": ["HistoCore"],
            "output_dir": "results/full_benchmark",
            "task_spec": {
                "dataset_name": "PatchCamelyon",
                "data_root": "data/pcam",
                "model_architecture": "resnet18_transformer",
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path)

            assert isinstance(config, BenchmarkConfig)
            assert config.mode == "full"
            assert config.frameworks == ["HistoCore"]
            assert config.output_dir == Path("results/full_benchmark")

        finally:
            config_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file(Path("nonexistent.json"))

    def test_load_config_unsupported_format(self):
        """Test loading configuration from unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                load_config_from_file(config_path)
        finally:
            config_path.unlink()


class TestConfigCreation:
    """Test configuration creation from CLI arguments."""

    def test_create_config_from_args_defaults(self):
        """Test creating config with default arguments."""
        parser = create_parser()
        args = parser.parse_args(["run"])

        config = create_config_from_args(args)

        assert isinstance(config, BenchmarkConfig)
        assert config.mode == "quick"
        assert config.frameworks == ["HistoCore", "PathML", "CLAM", "PyTorch"]
        assert config.quick_mode_epochs == 3
        assert config.quick_mode_samples == 1000
        assert config.random_seed == 42

        assert isinstance(config.task_spec, TaskSpecification)
        assert config.task_spec.dataset_name == "PatchCamelyon"
        assert config.task_spec.data_root == Path("data/pcam")
        assert config.task_spec.model_architecture == "resnet18_transformer"

    def test_create_config_from_args_custom(self):
        """Test creating config with custom arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--mode",
                "full",
                "--frameworks",
                "HistoCore",
                "PathML",
                "--output-dir",
                "custom_output",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.001",
                "--num-epochs",
                "20",
                "--random-seed",
                "123",
            ]
        )

        config = create_config_from_args(args)

        assert config.mode == "full"
        assert config.frameworks == ["HistoCore", "PathML"]
        assert config.output_dir == Path("custom_output")
        assert config.random_seed == 123

        assert config.task_spec.batch_size == 64
        assert config.task_spec.learning_rate == 0.001
        assert config.task_spec.num_epochs == 20
        assert config.task_spec.random_seed == 123


class TestListFrameworksCommand:
    """Test list-frameworks command."""

    def test_list_frameworks_command(self, capsys):
        """Test list-frameworks command output."""
        parser = create_parser()
        args = parser.parse_args(["list-frameworks"])

        exit_code = list_frameworks_command(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Available frameworks for benchmarking:" in captured.out
        assert "HistoCore" in captured.out
        assert "PathML" in captured.out
        assert "CLAM" in captured.out
        assert "PyTorch" in captured.out


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @patch("experiments.benchmark_system.cli.BenchmarkOrchestrator")
    def test_run_command_integration(self, mock_orchestrator_class):
        """Test run command integration with orchestrator."""
        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_result = MagicMock()
        mock_result.total_duration_hours = 3.5
        mock_result.successful_frameworks = ["HistoCore", "PathML"]
        mock_result.failed_frameworks = []
        mock_result.report_path = Path("results/report.md")
        mock_result.visualization_dir = Path("results/visualizations")

        mock_orchestrator.run_benchmark_suite.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Import and run command
        from experiments.benchmark_system.cli import run_command

        parser = create_parser()
        args = parser.parse_args(["run", "--mode", "quick"])

        exit_code = run_command(args)

        assert exit_code == 0
        mock_orchestrator_class.assert_called_once()
        mock_orchestrator.run_benchmark_suite.assert_called_once()

    @patch("experiments.benchmark_system.resource_manager.ResourceManager")
    def test_validate_command_integration(self, mock_resource_manager_class):
        """Test validate command integration with resource manager."""
        # Mock GPU info
        mock_gpu_info = MagicMock()
        mock_gpu_info.available = True
        mock_gpu_info.name = "NVIDIA RTX 4070"
        mock_gpu_info.memory_total_mb = 12288
        mock_gpu_info.cuda_available = True

        mock_resource_manager = MagicMock()
        mock_resource_manager.verify_gpu_availability.return_value = mock_gpu_info
        mock_resource_manager_class.return_value = mock_resource_manager

        # Import and run command
        from experiments.benchmark_system.cli import validate_command

        parser = create_parser()
        args = parser.parse_args(["validate"])

        exit_code = validate_command(args)

        assert exit_code == 0
        mock_resource_manager.verify_gpu_availability.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
