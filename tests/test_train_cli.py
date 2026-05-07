"""CLI regression tests for the training entrypoint."""

import subprocess
import sys
from pathlib import Path


def test_train_help_runs_on_python_314():
    """`python scripts/train.py --help` should render instead of crashing in Hydra."""
    repo_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "train is powered by Hydra." in result.stdout
    assert "Powered by Hydra" in result.stdout
    assert "Configuration groups" in result.stdout
