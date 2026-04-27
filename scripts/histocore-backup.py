#!/usr/bin/env python3
"""HistoCore backup and restore utility."""

import asyncio
import click
import json
import os
import shutil
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml


@click.group()
@click.option('--config', '-c', default='config/backup.yml', help='Backup config file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """HistoCore backup and restore utility."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['verbose'] = verbose


def load_config(config_path: str) -> Dict:
    """Load backup configuration."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'backup_dir': '/var/backups/histocore',
            'retention_days': 30,
            'components': {
                'prometheus': {
                    'data_dir': '/prometheus',
                    'enabled': True
                },
                'grafana': {
                    'data_dir': '/var/lib/grafana',
                    'enabled': True
                },
                'models': {
                    'data_dir': '/app/models',
                    'enabled': True
                },
                'configs': {
                    'data_dir': '/app/config',
                    'enabled': True
                }
            }
        }


@cli.command()
@click.option('--component', '-c', multiple=True, help='Specific components to backup')
@click.option('--compress', is_flag=True, default=True, help='Compress backup')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def backup(ctx, component, compress, output):
    """Create system backup."""
    config = ctx.obj['config']
    verbose = ctx.obj['verbose']
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not output:
        backup_dir = Path(config['backup_dir'])
        backup_dir.mkdir(parents=True, exist_ok=True)
        output = backup_dir / f"histocore_backup_{timestamp}.tar.gz"
    
    components_to_backup = component if component else config['components'].keys()
    
    click.echo(f"Creating backup: {output}")
    click.echo(f"Components: {', '.join(components_to_backup)}")
    
    temp_dir = Path(f"/tmp/histocore_backup_{timestamp}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Backup metadata
        metadata = {
            'timestamp': timestamp,
            'version': '1.0.0',
            'components': list(components_to_backup)
        }
        
        with open(temp_dir / 'backup_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Backup each component
        for comp_name in components_to_backup:
            if comp_name not in config['components']:
                click.echo(f"Warning: Unknown component {comp_name}")
                continue
                
            comp_config = config['components'][comp_name]
            if not comp_config.get('enabled', True):
                click.echo(f"Skipping disabled component: {comp_name}")
                continue
                
            click.echo(f"Backing up {comp_name}...")
            
            if comp_name == 'prometheus':
                backup_prometheus(comp_config, temp_dir / comp_name, verbose)
            elif comp_name == 'grafana':
                backup_grafana(comp_config, temp_dir / comp_name, verbose)
            elif comp_name == 'models':
                backup_models(comp_config, temp_dir / comp_name, verbose)
            elif comp_name == 'configs':
                backup_configs(comp_config, temp_dir / comp_name, verbose)
            else:
                backup_directory(comp_config, temp_dir / comp_name, verbose)
                
        # Create archive
        click.echo("Creating archive...")
        
        if compress:
            with tarfile.open(output, 'w:gz') as tar:
                tar.add(temp_dir, arcname='.')
        else:
            with tarfile.open(output, 'w') as tar:
                tar.add(temp_dir, arcname='.')
                
        # Get file size
        size_mb = os.path.getsize(output) / (1024 * 1024)
        
        click.echo(f"✓ Backup created: {output} ({size_mb:.1f} MB)")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def backup_prometheus(config: Dict, output_dir: Path, verbose: bool):
    """Backup Prometheus data."""
    data_dir = Path(config['data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        click.echo(f"Warning: Prometheus data directory not found: {data_dir}")
        return
        
    # Copy TSDB data
    if (data_dir / 'data').exists():
        shutil.copytree(data_dir / 'data', output_dir / 'data')
        
    # Copy configuration
    if (data_dir / 'prometheus.yml').exists():
        shutil.copy2(data_dir / 'prometheus.yml', output_dir)
        
    # Copy alert rules
    if (data_dir / 'alerts').exists():
        shutil.copytree(data_dir / 'alerts', output_dir / 'alerts')
        
    if verbose:
        click.echo(f"  Prometheus data backed up to {output_dir}")


def backup_grafana(config: Dict, output_dir: Path, verbose: bool):
    """Backup Grafana data."""
    data_dir = Path(config['data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        click.echo(f"Warning: Grafana data directory not found: {data_dir}")
        return
        
    # Copy database
    if (data_dir / 'grafana.db').exists():
        shutil.copy2(data_dir / 'grafana.db', output_dir)
        
    # Copy dashboards
    if (data_dir / 'dashboards').exists():
        shutil.copytree(data_dir / 'dashboards', output_dir / 'dashboards')
        
    # Copy provisioning
    if (data_dir / 'provisioning').exists():
        shutil.copytree(data_dir / 'provisioning', output_dir / 'provisioning')
        
    if verbose:
        click.echo(f"  Grafana data backed up to {output_dir}")


def backup_models(config: Dict, output_dir: Path, verbose: bool):
    """Backup ML models."""
    data_dir = Path(config['data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        click.echo(f"Warning: Models directory not found: {data_dir}")
        return
        
    # Copy model files
    for model_file in data_dir.glob('*.pth'):
        shutil.copy2(model_file, output_dir)
        
    for model_file in data_dir.glob('*.onnx'):
        shutil.copy2(model_file, output_dir)
        
    # Copy model metadata
    if (data_dir / 'model_registry.json').exists():
        shutil.copy2(data_dir / 'model_registry.json', output_dir)
        
    if verbose:
        click.echo(f"  Models backed up to {output_dir}")


def backup_configs(config: Dict, output_dir: Path, verbose: bool):
    """Backup configuration files."""
    data_dir = Path(config['data_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        click.echo(f"Warning: Config directory not found: {data_dir}")
        return
        
    # Copy all config files
    for config_file in data_dir.glob('*.yml'):
        shutil.copy2(config_file, output_dir)
        
    for config_file in data_dir.glob('*.yaml'):
        shutil.copy2(config_file, output_dir)
        
    for config_file in data_dir.glob('*.json'):
        shutil.copy2(config_file, output_dir)
        
    if verbose:
        click.echo(f"  Configs backed up to {output_dir}")


def backup_directory(config: Dict, output_dir: Path, verbose: bool):
    """Backup generic directory."""
    data_dir = Path(config['data_dir'])
    
    if not data_dir.exists():
        click.echo(f"Warning: Directory not found: {data_dir}")
        return
        
    shutil.copytree(data_dir, output_dir)
    
    if verbose:
        click.echo(f"  Directory backed up to {output_dir}")


@cli.command()
@click.argument('backup_file')
@click.option('--component', '-c', multiple=True, help='Specific components to restore')
@click.option('--dry-run', is_flag=True, help='Show what would be restored')
@click.option('--force', is_flag=True, help='Overwrite existing data')
@click.pass_context
def restore(ctx, backup_file, component, dry_run, force):
    """Restore from backup."""
    config = ctx.obj['config']
    verbose = ctx.obj['verbose']
    
    backup_path = Path(backup_file)
    if not backup_path.exists():
        click.echo(f"✗ Backup file not found: {backup_file}", err=True)
        return
        
    click.echo(f"Restoring from: {backup_file}")
    
    temp_dir = Path(f"/tmp/histocore_restore_{int(time.time())}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract backup
        click.echo("Extracting backup...")
        with tarfile.open(backup_path, 'r:*') as tar:
            tar.extractall(temp_dir)
            
        # Read metadata
        metadata_file = temp_dir / 'backup_metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                
            click.echo(f"Backup timestamp: {metadata['timestamp']}")
            click.echo(f"Backup version: {metadata['version']}")
            click.echo(f"Available components: {', '.join(metadata['components'])}")
        else:
            click.echo("Warning: No metadata found in backup")
            metadata = {'components': []}
            
        # Determine components to restore
        components_to_restore = component if component else metadata.get('components', [])
        
        if dry_run:
            click.echo("\nDry run - would restore:")
            for comp_name in components_to_restore:
                comp_dir = temp_dir / comp_name
                if comp_dir.exists():
                    click.echo(f"  {comp_name}: {comp_dir}")
            return
            
        # Restore components
        for comp_name in components_to_restore:
            comp_dir = temp_dir / comp_name
            if not comp_dir.exists():
                click.echo(f"Warning: Component {comp_name} not found in backup")
                continue
                
            if comp_name not in config['components']:
                click.echo(f"Warning: Unknown component {comp_name}")
                continue
                
            comp_config = config['components'][comp_name]
            target_dir = Path(comp_config['data_dir'])
            
            click.echo(f"Restoring {comp_name} to {target_dir}...")
            
            if target_dir.exists() and not force:
                click.echo(f"  Target exists, use --force to overwrite")
                continue
                
            if target_dir.exists():
                shutil.rmtree(target_dir)
                
            shutil.copytree(comp_dir, target_dir)
            
            if verbose:
                click.echo(f"  Restored {comp_name}")
                
        click.echo("✓ Restore completed")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@cli.command()
@click.pass_context
def list_backups(ctx):
    """List available backups."""
    config = ctx.obj['config']
    backup_dir = Path(config['backup_dir'])
    
    if not backup_dir.exists():
        click.echo("No backup directory found")
        return
        
    backups = list(backup_dir.glob('histocore_backup_*.tar.gz'))
    backups.extend(backup_dir.glob('histocore_backup_*.tar'))
    
    if not backups:
        click.echo("No backups found")
        return
        
    backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    headers = ['Backup File', 'Size', 'Created', 'Age']
    rows = []
    
    for backup in backups:
        stat = backup.stat()
        size_mb = stat.st_size / (1024 * 1024)
        created = datetime.fromtimestamp(stat.st_mtime)
        age_days = (datetime.now() - created).days
        
        rows.append([
            backup.name,
            f"{size_mb:.1f} MB",
            created.strftime('%Y-%m-%d %H:%M'),
            f"{age_days} days"
        ])
        
    from tabulate import tabulate
    click.echo(tabulate(rows, headers=headers, tablefmt='grid'))


@cli.command()
@click.option('--days', '-d', default=None, type=int, help='Delete backups older than N days')
@click.option('--keep', '-k', default=5, type=int, help='Keep N most recent backups')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted')
@click.pass_context
def cleanup(ctx, days, keep, dry_run):
    """Clean up old backups."""
    config = ctx.obj['config']
    backup_dir = Path(config['backup_dir'])
    
    if not backup_dir.exists():
        click.echo("No backup directory found")
        return
        
    # Use config default if not specified
    if days is None:
        days = config.get('retention_days', 30)
        
    backups = list(backup_dir.glob('histocore_backup_*.tar.gz'))
    backups.extend(backup_dir.glob('histocore_backup_*.tar'))
    
    if not backups:
        click.echo("No backups found")
        return
        
    backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep most recent N backups
    to_keep = set(backups[:keep])
    
    # Delete old backups
    cutoff_time = time.time() - (days * 24 * 3600)
    to_delete = []
    
    for backup in backups:
        if backup not in to_keep and backup.stat().st_mtime < cutoff_time:
            to_delete.append(backup)
            
    if not to_delete:
        click.echo("No backups to delete")
        return
        
    if dry_run:
        click.echo("Would delete:")
        for backup in to_delete:
            click.echo(f"  {backup.name}")
        return
        
    click.echo(f"Deleting {len(to_delete)} old backups...")
    
    for backup in to_delete:
        backup.unlink()
        click.echo(f"  Deleted {backup.name}")
        
    click.echo("✓ Cleanup completed")


@cli.command()
@click.argument('backup_file')
@click.pass_context
def info(ctx, backup_file):
    """Show backup information."""
    backup_path = Path(backup_file)
    if not backup_path.exists():
        click.echo(f"✗ Backup file not found: {backup_file}", err=True)
        return
        
    temp_dir = Path(f"/tmp/histocore_info_{int(time.time())}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract metadata only
        with tarfile.open(backup_path, 'r:*') as tar:
            try:
                metadata_member = tar.getmember('backup_metadata.json')
                tar.extract(metadata_member, temp_dir)
                
                with open(temp_dir / 'backup_metadata.json') as f:
                    metadata = json.load(f)
                    
                click.echo(f"Backup File: {backup_file}")
                click.echo(f"Timestamp: {metadata['timestamp']}")
                click.echo(f"Version: {metadata['version']}")
                click.echo(f"Components: {', '.join(metadata['components'])}")
                
                # File info
                stat = backup_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                created = datetime.fromtimestamp(stat.st_mtime)
                
                click.echo(f"File Size: {size_mb:.1f} MB")
                click.echo(f"Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
                
            except KeyError:
                click.echo("No metadata found in backup")
                
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    cli()