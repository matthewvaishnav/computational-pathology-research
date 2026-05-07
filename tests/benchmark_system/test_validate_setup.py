"""
Unit tests for validate_setup.py script.

Tests the setup validation functionality for the Competitor Benchmark System.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_validate_setup_help():
    """Test that validate_setup.py --help works."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Validate setup for Competitor Benchmark System" in result.stdout
    assert "--output" in result.stdout
    assert "--verbose" in result.stdout


def test_validate_setup_runs():
    """Test that validate_setup.py runs without crashing."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py"],
        capture_output=True,
        text=True,
    )
    
    # Script should run (may pass or fail depending on system)
    assert result.returncode in [0, 1]
    assert "Competitor Benchmark System - Setup Validation" in result.stdout
    assert "Validation Summary" in result.stdout


def test_validate_setup_json_output(tmp_path):
    """Test that validate_setup.py can save JSON report."""
    output_file = tmp_path / "validation_report.json"
    
    result = subprocess.run(
        [
            sys.executable,
            "experiments/benchmark_system/validate_setup.py",
            "--output",
            str(output_file),
        ],
        capture_output=True,
        text=True,
    )
    
    # Script should run
    assert result.returncode in [0, 1]
    
    # JSON file should be created
    assert output_file.exists()
    
    # JSON should be valid
    with open(output_file, "r") as f:
        report = json.load(f)
    
    # Verify report structure
    assert "timestamp" in report
    assert "overall_status" in report
    assert "checks" in report
    assert "summary" in report
    
    # Verify summary
    assert "total_checks" in report["summary"]
    assert "passed" in report["summary"]
    assert "failed" in report["summary"]
    assert "warnings" in report["summary"]
    
    # Verify checks
    assert len(report["checks"]) > 0
    for check in report["checks"]:
        assert "check_name" in check
        assert "passed" in check
        assert "message" in check
        assert "details" in check
        assert "warnings" in check


def test_validate_setup_checks_gpu():
    """Test that validate_setup.py checks GPU availability."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py"],
        capture_output=True,
        text=True,
    )
    
    assert "Checking GPU availability" in result.stdout


def test_validate_setup_checks_cuda():
    """Test that validate_setup.py checks CUDA and cuDNN."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py"],
        capture_output=True,
        text=True,
    )
    
    assert "Checking CUDA and cuDNN versions" in result.stdout


def test_validate_setup_checks_disk_space():
    """Test that validate_setup.py checks disk space."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py"],
        capture_output=True,
        text=True,
    )
    
    assert "Checking disk space availability" in result.stdout


def test_validate_setup_checks_python():
    """Test that validate_setup.py checks Python compatibility."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py"],
        capture_output=True,
        text=True,
    )
    
    assert "Checking Python version compatibility" in result.stdout


def test_validate_setup_checks_framework_imports():
    """Test that validate_setup.py checks framework imports."""
    result = subprocess.run(
        [sys.executable, "experiments/benchmark_system/validate_setup.py"],
        capture_output=True,
        text=True,
    )
    
    assert "Running smoke tests for framework imports" in result.stdout
    assert "PyTorch" in result.stdout
