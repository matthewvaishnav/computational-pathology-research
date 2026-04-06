"""
Experiment tracking module.
Records experiment metadata (git hash, timestamp, config, metrics) to results/experiments/.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import Any

_EXPERIMENTS_DIR = 'results/experiments'


def _get_git_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _get_git_branch() -> str:
    """Get the current git branch."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _ensure_experiments_dir() -> None:
    """Ensure the experiments directory exists."""
    os.makedirs(_EXPERIMENTS_DIR, exist_ok=True)


def log_experiment(name: str, config: dict[str, Any], metrics: dict[str, Any]) -> str:
    """
    Log a single experiment run.

    Args:
        name: Experiment name (e.g., 'quick_demo', 'missing_modality_config_1')
        config: Configuration dictionary with model/data hyperparameters
        metrics: Metrics dictionary with performance values

    Returns:
        Path to the saved experiment JSON file
    """
    _ensure_experiments_dir()

    timestamp = datetime.now().isoformat()
    git_hash = _get_git_hash()
    git_branch = _get_git_branch()

    experiment = {
        'name': name,
        'timestamp': timestamp,
        'git_hash': git_hash,
        'git_branch': git_branch,
        'config': config,
        'metrics': metrics
    }

    # Create safe filename from name and timestamp
    safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{safe_name}_{timestamp_str}.json'
    filepath = os.path.join(_EXPERIMENTS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(experiment, f, indent=2)

    return filepath


def get_experiment_history() -> list[dict[str, Any]]:
    """
    Get the history of all logged experiments.

    Returns:
        List of experiment dictionaries sorted by timestamp (newest first)
    """
    _ensure_experiments_dir()

    experiments = []
    for filename in os.listdir(_EXPERIMENTS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(_EXPERIMENTS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    experiments.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue

    # Sort by timestamp, newest first
    experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return experiments


def get_experiment_by_name(name: str) -> list[dict[str, Any]]:
    """
    Get all experiments with a specific name.

    Args:
        name: Experiment name to filter by

    Returns:
        List of experiment dictionaries matching the name
    """
    history = get_experiment_history()
    return [exp for exp in history if exp.get('name') == name]


def get_latest_experiment(name: str) -> dict[str, Any] | None:
    """
    Get the most recent experiment with a specific name.

    Args:
        name: Experiment name

    Returns:
        The most recent experiment dict, or None if not found
    """
    matches = get_experiment_by_name(name)
    return matches[0] if matches else None
