"""
Experiment registry.
Auto-generated registry of all experiments with links to their results.
"""
import json
import os
import re
from glob import glob
from typing import Any

_REGISTRY_FILE = 'results/experiments/registry.json'
_EXPERIMENTS_DIR = 'results/experiments'


def _get_readable_size(filepath: str) -> str:
    """Get human-readable file size."""
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} TB'


def _get_modification_time(filepath: str) -> str:
    """Get modification time in readable format."""
    mtime = os.path.getmtime(filepath)
    from datetime import datetime
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def _scan_experiments() -> list[dict[str, Any]]:
    """Scan the experiments directory and build registry entries."""
    experiments = []

    pattern = os.path.join(_EXPERIMENTS_DIR, '*.json')
    for filepath in glob(pattern):
        filename = os.path.basename(filepath)

        # Skip registry itself
        if filename == 'registry.json':
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            entry = {
                'name': data.get('name', 'unknown'),
                'timestamp': data.get('timestamp', ''),
                'git_hash': data.get('git_hash', 'unknown')[:8],
                'git_branch': data.get('git_branch', 'unknown'),
                'config': data.get('config', {}),
                'metrics': data.get('metrics', {}),
                'result_file': filename,
                'size': _get_readable_size(filepath),
                'modified': _get_modification_time(filepath)
            }
            experiments.append(entry)
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by timestamp, newest first
    experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return experiments


def update_registry() -> list[dict[str, Any]]:
    """
    Update the experiment registry by scanning the results/experiments/ directory.

    Returns:
        List of all experiment entries
    """
    os.makedirs(_EXPERIMENTS_DIR, exist_ok=True)

    experiments = _scan_experiments()

    registry = {
        'updated': __import__('datetime').datetime.now().isoformat(),
        'total_experiments': len(experiments),
        'experiments': experiments
    }

    with open(_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)

    return experiments


def get_registry() -> list[dict[str, Any]]:
    """
    Get the current experiment registry.
    If registry doesn't exist, updates it first.

    Returns:
        List of experiment entries
    """
    if os.path.exists(_REGISTRY_FILE):
        try:
            with open(_REGISTRY_FILE, 'r') as f:
                registry = json.load(f)
            return registry.get('experiments', [])
        except (json.JSONDecodeError, IOError):
            pass

    return update_registry()


def generate_readme() -> str:
    """
    Generate a README.md for the experiments directory.

    Returns:
        Markdown-formatted string with experiment summary
    """
    registry = get_registry()

    lines = [
        '# Experiment Registry',
        '',
        f'Auto-generated on: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        f'Total experiments: {len(registry)}',
        '',
        '## Experiments',
        ''
    ]

    if not registry:
        lines.append('_No experiments recorded yet._')
        lines.append('')
        lines.append('Experiments are logged automatically by calling `log_experiment()` from your training scripts.')
    else:
        # Group by name
        from collections import defaultdict
        by_name = defaultdict(list)
        for exp in registry:
            by_name[exp['name']].append(exp)

        for name, exps in by_name.items():
            latest = exps[0]  # Already sorted newest first
            lines.append(f'### {name}')
            lines.append('')
            lines.append(f'- Latest run: {latest["timestamp"]}')
            lines.append(f'- Git commit: `{latest["git_hash"]}`')
            lines.append(f'- Branch: `{latest["git_branch"]}`')
            lines.append(f'- Metrics: {json.dumps(latest["metrics"])}')
            lines.append(f'- Result file: `{latest["result_file"]}`')
            lines.append(f'- All runs: {len(exps)}')
            lines.append('')

    return '\n'.join(lines)


def update_readme() -> None:
    """Update the README.md in the experiments directory."""
    readme_path = os.path.join(_EXPERIMENTS_DIR, 'README.md')
    content = generate_readme()
    with open(readme_path, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    update_registry()
    update_readme()
    print(f'Registry updated. Found {len(get_registry())} experiments.')
