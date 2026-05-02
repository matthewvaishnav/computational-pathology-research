#!/usr/bin/env python3
"""HistoCore administration CLI tool."""

import asyncio
import click
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import yaml
from tabulate import tabulate


@click.group()
@click.option('--config', '-c', default='config/admin.yml', help='Config file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """HistoCore streaming administration tool."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['verbose'] = verbose


def load_config(config_path: str) -> Dict[str, Any]:
    """Load admin configuration."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'endpoints': {
                'streaming': 'http://localhost:8000',
                'metrics': 'http://localhost:9090',
                'health': 'http://localhost:8080'
            }
        }


@cli.group()
def health():
    """Health check commands."""
    pass


@health.command()
@click.pass_context
def status(ctx):
    """Check system health status."""
    config = ctx.obj['config']
    health_url = config['endpoints']['health']
    
    try:
        response = requests.get(f"{health_url}/health/detailed", timeout=10, timeout=30)
        data = response.json()
        
        # Overall status
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow', 
            'unhealthy': 'red'
        }
        
        click.echo(f"Overall Status: {click.style(data['status'].upper(), fg=status_color.get(data['status'], 'white'))}")
        click.echo(f"Message: {data['message']}")
        click.echo(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))}")
        click.echo()
        
        # Component details
        headers = ['Component', 'Status', 'Message', 'Response Time']
        rows = []
        
        for component in data['components']:
            status_symbol = {
                'healthy': '✓',
                'degraded': '⚠',
                'unhealthy': '✗'
            }
            
            rows.append([
                component['name'],
                f"{status_symbol.get(component['status'], '?')} {component['status']}",
                component['message'][:50] + ('...' if len(component['message']) > 50 else ''),
                f"{component['response_time_ms']:.1f}ms"
            ])
            
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        
        # Summary
        summary = data['summary']
        click.echo(f"\nSummary:")
        click.echo(f"  Total: {summary['total_components']}")
        click.echo(f"  Healthy: {summary['healthy_components']}")
        click.echo(f"  Degraded: {summary['degraded_components']}")
        click.echo(f"  Unhealthy: {summary['unhealthy_components']}")
        
    except Exception as e:
        click.echo(f"Error checking health: {e}", err=True)
        sys.exit(1)


@health.command()
@click.pass_context
def live(ctx):
    """Check liveness probe."""
    config = ctx.obj['config']
    health_url = config['endpoints']['health']
    
    try:
        response = requests.get(f"{health_url}/health/live", timeout=5, timeout=30)
        if response.status_code == 200:
            click.echo("✓ Service is alive")
        else:
            click.echo(f"✗ Service not responding: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Liveness check failed: {e}")
        sys.exit(1)


@health.command()
@click.pass_context
def ready(ctx):
    """Check readiness probe."""
    config = ctx.obj['config']
    health_url = config['endpoints']['health']
    
    try:
        response = requests.get(f"{health_url}/health/ready", timeout=5, timeout=30)
        if response.status_code == 200:
            click.echo("✓ Service is ready")
        else:
            click.echo(f"✗ Service not ready: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Readiness check failed: {e}")
        sys.exit(1)


@cli.group()
def metrics():
    """Metrics commands."""
    pass


@metrics.command()
@click.option('--query', '-q', required=True, help='Prometheus query')
@click.option('--time', '-t', help='Query time (RFC3339 or Unix timestamp)')
@click.pass_context
def query(ctx, query, time):
    """Execute Prometheus query."""
    config = ctx.obj['config']
    metrics_url = config['endpoints']['metrics']
    
    params = {'query': query}
    if time:
        params['time'] = time
        
    try:
        response = requests.get(f"{metrics_url}/api/v1/query", params=params, timeout=10, timeout=30)
        data = response.json()
        
        if data['status'] == 'success':
            results = data['data']['result']
            
            if not results:
                click.echo("No data found")
                return
                
            headers = ['Metric', 'Value', 'Timestamp']
            rows = []
            
            for result in results:
                metric_name = result['metric'].get('__name__', 'unknown')
                labels = ', '.join([f"{k}={v}" for k, v in result['metric'].items() if k != '__name__'])
                if labels:
                    metric_name += f"{{{labels}}}"
                    
                value = result['value'][1]
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(result['value'][0])))
                
                rows.append([metric_name, value, timestamp])
                
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
            
        else:
            click.echo(f"Query failed: {data.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error executing query: {e}", err=True)
        sys.exit(1)


@metrics.command()
@click.pass_context
def summary(ctx):
    """Show key metrics summary."""
    config = ctx.obj['config']
    metrics_url = config['endpoints']['metrics']
    
    queries = {
        'Slides Processed (5m)': 'rate(histocore_slides_processed_total[5m])',
        'Processing Time (95th)': 'histogram_quantile(0.95, rate(histocore_processing_duration_seconds_bucket[5m]))',
        'GPU Memory Usage': 'histocore_gpu_memory_usage_bytes{type="used"} / histocore_gpu_memory_usage_bytes{type="total"} * 100',
        'Error Rate (5m)': 'rate(histocore_errors_total[5m])',
        'Throughput': 'histocore_throughput_patches_per_second'
    }
    
    results = {}
    for name, query in queries.items():
        try:
            response = requests.get(f"{metrics_url}/api/v1/query", params={'query': query}, timeout=5, timeout=30)
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                value = data['data']['result'][0]['value'][1]
                results[name] = f"{float(value):.2f}"
            else:
                results[name] = "N/A"
        except Exception as e:
            results[name] = "Error"
            print(f"Check failed for {name}: {type(e).__name__}")
            
    headers = ['Metric', 'Value']
    rows = [[k, v] for k, v in results.items()]
    
    click.echo("Key Metrics Summary:")
    click.echo(tabulate(rows, headers=headers, tablefmt='grid'))


@cli.group()
def processing():
    """Processing management commands."""
    pass


@processing.command()
@click.argument('slide_path')
@click.option('--priority', '-p', type=int, default=1, help='Processing priority')
@click.option('--wait', '-w', is_flag=True, help='Wait for completion')
@click.pass_context
def submit(ctx, slide_path, priority, wait):
    """Submit slide for processing."""
    config = ctx.obj['config']
    streaming_url = config['endpoints']['streaming']
    
    payload = {
        'slide_path': slide_path,
        'priority': priority
    }
    
    try:
        response = requests.post(f"{streaming_url}/api/v1/process", json=payload, timeout=10, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get('job_id')
            click.echo(f"✓ Slide submitted successfully")
            click.echo(f"Job ID: {job_id}")
            
            if wait and job_id:
                click.echo("Waiting for completion...")
                wait_for_job(ctx, job_id)
                
        else:
            click.echo(f"✗ Submission failed: {response.status_code} {response.text}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Error submitting slide: {e}", err=True)
        sys.exit(1)


@processing.command()
@click.argument('job_id')
@click.pass_context
def status_job(ctx, job_id):
    """Check job status."""
    config = ctx.obj['config']
    streaming_url = config['endpoints']['streaming']
    
    try:
        response = requests.get(f"{streaming_url}/api/v1/jobs/{job_id}", timeout=10, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            click.echo(f"Job ID: {data['job_id']}")
            click.echo(f"Status: {data['status']}")
            click.echo(f"Progress: {data.get('progress', 0):.1f}%")
            click.echo(f"Created: {data['created_at']}")
            
            if data.get('completed_at'):
                click.echo(f"Completed: {data['completed_at']}")
                
            if data.get('error'):
                click.echo(f"Error: {data['error']}")
                
        else:
            click.echo(f"✗ Job not found: {response.status_code}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Error checking job: {e}", err=True)
        sys.exit(1)


@processing.command()
@click.pass_context
def queue(ctx):
    """Show processing queue."""
    config = ctx.obj['config']
    streaming_url = config['endpoints']['streaming']
    
    try:
        response = requests.get(f"{streaming_url}/api/v1/queue", timeout=10, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if not data['jobs']:
                click.echo("Queue is empty")
                return
                
            headers = ['Job ID', 'Status', 'Priority', 'Progress', 'Created']
            rows = []
            
            for job in data['jobs']:
                rows.append([
                    job['job_id'][:8] + '...',
                    job['status'],
                    job.get('priority', 1),
                    f"{job.get('progress', 0):.1f}%",
                    job['created_at']
                ])
                
            click.echo(f"Processing Queue ({len(data['jobs'])} jobs):")
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
            
        else:
            click.echo(f"✗ Error getting queue: {response.status_code}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Error getting queue: {e}", err=True)
        sys.exit(1)


def wait_for_job(ctx, job_id):
    """Wait for job completion."""
    config = ctx.obj['config']
    streaming_url = config['endpoints']['streaming']
    
    while True:
        try:
            response = requests.get(f"{streaming_url}/api/v1/jobs/{job_id}", timeout=10, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                status = data['status']
                progress = data.get('progress', 0)
                
                click.echo(f"\rProgress: {progress:.1f}% ({status})", nl=False)
                
                if status in ['completed', 'failed', 'cancelled']:
                    click.echo()
                    if status == 'completed':
                        click.echo("✓ Job completed successfully")
                    else:
                        click.echo(f"✗ Job {status}")
                        if data.get('error'):
                            click.echo(f"Error: {data['error']}")
                    break
                    
            time.sleep(2)
            
        except KeyboardInterrupt:
            click.echo("\nWait cancelled")
            break
        except Exception as e:
            click.echo(f"\nError waiting for job: {e}")
            break


@cli.group()
def config():
    """Configuration management."""
    pass


@config.command()
@click.pass_context
def show(ctx):
    """Show current configuration."""
    config = ctx.obj['config']
    click.echo(yaml.dump(config, default_flow_style=False))


@config.command()
@click.argument('key')
@click.argument('value')
@click.pass_context
def set_config(ctx, key, value):
    """Set configuration value."""
    # This would update runtime config via API
    config = ctx.obj['config']
    streaming_url = config['endpoints']['streaming']
    
    payload = {key: value}
    
    try:
        response = requests.post(f"{streaming_url}/api/v1/config", json=payload, timeout=10, timeout=30)
        
        if response.status_code == 200:
            click.echo(f"✓ Configuration updated: {key} = {value}")
        else:
            click.echo(f"✗ Failed to update config: {response.status_code}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Error updating config: {e}", err=True)
        sys.exit(1)


@cli.group()
def logs():
    """Log management commands."""
    pass


@logs.command()
@click.option('--lines', '-n', default=100, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--level', '-l', help='Filter by log level')
@click.option('--component', '-c', help='Filter by component')
@click.pass_context
def tail(ctx, lines, follow, level, component):
    """Tail application logs."""
    config = ctx.obj['config']
    streaming_url = config['endpoints']['streaming']
    
    params = {'lines': lines}
    if level:
        params['level'] = level
    if component:
        params['component'] = component
        
    try:
        response = requests.get(f"{streaming_url}/api/v1/logs", params=params, timeout=10, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            for log_entry in data['logs']:
                timestamp = log_entry.get('timestamp', '')
                level = log_entry.get('level', 'INFO')
                component = log_entry.get('component', 'unknown')
                message = log_entry.get('message', '')
                
                level_colors = {
                    'DEBUG': 'blue',
                    'INFO': 'green', 
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red'
                }
                
                level_colored = click.style(level, fg=level_colors.get(level, 'white'))
                click.echo(f"{timestamp} [{level_colored}] {component}: {message}")
                
            if follow:
                # In real implementation, this would use WebSocket or SSE
                click.echo("Follow mode not implemented in demo")
                
        else:
            click.echo(f"✗ Error getting logs: {response.status_code}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Error getting logs: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def dashboard(ctx):
    """Open monitoring dashboard."""
    config = ctx.obj['config']
    
    # Get Grafana URL from config or default
    grafana_url = config.get('endpoints', {}).get('grafana', 'http://localhost:3000')
    
    click.echo(f"Opening dashboard: {grafana_url}")
    
    import webbrowser
    webbrowser.open(grafana_url)


if __name__ == '__main__':
    cli()