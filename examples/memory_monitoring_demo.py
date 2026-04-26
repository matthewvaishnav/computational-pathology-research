"""Demo script for memory monitoring and alerting system.

This script demonstrates:
- Real-time memory usage tracking
- Memory pressure detection and alerting
- Memory usage analytics and reporting
- Integration with GPU pipeline
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
import sys
import importlib.util

# Load memory_optimizer module directly
memory_optimizer_path = Path(__file__).parent.parent / "src" / "streaming" / "memory_optimizer.py"
spec = importlib.util.spec_from_file_location("memory_optimizer", memory_optimizer_path)
memory_optimizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_optimizer_module)

# Import classes from loaded module
MemoryMonitor = memory_optimizer_module.MemoryMonitor
MemoryPressureLevel = memory_optimizer_module.MemoryPressureLevel


def alert_callback(alert):
    """Callback function for memory alerts."""
    print(f"\n🚨 ALERT [{alert.severity.upper()}]: {alert.message}")
    print(f"   Current usage: {alert.current_usage_gb:.2f}GB / {alert.threshold_gb:.2f}GB")
    print(f"   Recommended action: {alert.recommended_action}\n")


def demo_basic_monitoring():
    """Demonstrate basic memory monitoring."""
    print("=" * 80)
    print("Demo 1: Basic Memory Monitoring")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create monitor
    monitor = MemoryMonitor(
        device=device,
        memory_limit_gb=2.0,
        sampling_interval_ms=100.0,
        enable_alerts=True,
        alert_callback=alert_callback
    )
    
    # Start monitoring
    print("\n📊 Starting memory monitoring...")
    monitor.start_monitoring()
    
    # Simulate workload
    print("🔄 Simulating workload...")
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)
        time.sleep(0.2)
        
        # Get current snapshot
        snapshot = monitor.get_current_snapshot()
        print(f"   Step {i+1}: {snapshot.allocated_gb:.3f}GB allocated, "
              f"pressure: {snapshot.pressure_level.value}")
    
    # Get analytics
    print("\n📈 Memory Analytics:")
    analytics = monitor.get_analytics()
    print(f"   Monitoring duration: {analytics.monitoring_duration_seconds:.1f}s")
    print(f"   Total snapshots: {analytics.total_snapshots}")
    print(f"   Peak usage: {analytics.peak_usage_gb:.3f}GB")
    print(f"   Average usage: {analytics.avg_usage_gb:.3f}GB")
    print(f"   Min usage: {analytics.min_usage_gb:.3f}GB")
    print(f"   Alerts triggered: {analytics.alerts_triggered}")
    
    # Pressure distribution
    print("\n📊 Pressure Distribution:")
    for level, percent in analytics.pressure_distribution.items():
        print(f"   {level}: {percent:.1f}%")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\n✅ Monitoring stopped")
    
    # Cleanup
    del tensors
    monitor.cleanup()


def demo_pressure_detection():
    """Demonstrate memory pressure detection."""
    print("\n" + "=" * 80)
    print("Demo 2: Memory Pressure Detection")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create monitor with low limit to trigger alerts
    monitor = MemoryMonitor(
        device=device,
        memory_limit_gb=0.5,  # Low limit for demo
        sampling_interval_ms=50.0,
        enable_alerts=True,
        alert_callback=alert_callback
    )
    
    # Customize thresholds
    monitor.set_pressure_threshold(MemoryPressureLevel.MODERATE, 0.50)
    monitor.set_pressure_threshold(MemoryPressureLevel.HIGH, 0.70)
    monitor.set_pressure_threshold(MemoryPressureLevel.CRITICAL, 0.85)
    
    print("\n📊 Custom pressure thresholds:")
    print(f"   MODERATE: 50%")
    print(f"   HIGH: 70%")
    print(f"   CRITICAL: 85%")
    
    # Start monitoring
    print("\n📊 Starting monitoring with pressure detection...")
    monitor.start_monitoring()
    
    # Gradually increase memory usage
    print("🔄 Gradually increasing memory usage...")
    tensors = []
    for i in range(20):
        tensor = torch.randn(500, 500, device=device)
        tensors.append(tensor)
        time.sleep(0.1)
        
        snapshot = monitor.get_current_snapshot()
        if i % 5 == 0:
            print(f"   Step {i+1}: {snapshot.utilization_percent:.1f}% usage, "
                  f"pressure: {snapshot.pressure_level.value}")
    
    # Wait for final snapshots
    time.sleep(0.3)
    
    # Get recent alerts
    print("\n🚨 Recent Alerts:")
    recent_alerts = monitor.get_recent_alerts(count=5)
    if recent_alerts:
        for alert in recent_alerts:
            print(f"   [{alert.severity}] {alert.message}")
    else:
        print("   No alerts triggered")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Cleanup
    del tensors
    monitor.cleanup()


def demo_analytics_reporting():
    """Demonstrate analytics and reporting."""
    print("\n" + "=" * 80)
    print("Demo 3: Analytics and Reporting")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create monitor
    monitor = MemoryMonitor(
        device=device,
        memory_limit_gb=2.0,
        sampling_interval_ms=100.0,
        enable_alerts=True
    )
    
    # Use context manager
    print("\n📊 Running workload with monitoring...")
    with monitor:
        # Simulate varying workload
        for cycle in range(3):
            print(f"\n   Cycle {cycle + 1}/3")
            tensors = []
            
            # Allocate
            for i in range(5):
                tensor = torch.randn(1000, 1000, device=device)
                tensors.append(tensor)
                time.sleep(0.1)
            
            # Deallocate
            del tensors
            time.sleep(0.2)
    
    # Generate comprehensive report
    print("\n📋 Comprehensive Memory Report:")
    print("=" * 80)
    
    report = monitor.generate_report()
    
    # Current status
    print("\n📊 Current Status:")
    status = report['current_status']
    print(f"   Allocated: {status['allocated_gb']:.3f}GB")
    print(f"   Reserved: {status['reserved_gb']:.3f}GB")
    print(f"   Total: {status['total_gb']:.3f}GB")
    print(f"   Utilization: {status['utilization_percent']:.1f}%")
    print(f"   Pressure: {status['pressure_level']}")
    
    # Analytics
    print("\n📈 Analytics:")
    analytics = report['analytics']
    print(f"   Duration: {analytics['monitoring_duration_seconds']:.1f}s")
    print(f"   Snapshots: {analytics['total_snapshots']}")
    print(f"   Peak: {analytics['peak_usage_gb']:.3f}GB")
    print(f"   Average: {analytics['avg_usage_gb']:.3f}GB")
    print(f"   Min: {analytics['min_usage_gb']:.3f}GB")
    
    # Pressure distribution
    print("\n📊 Pressure Distribution:")
    for level, percent in analytics['pressure_distribution'].items():
        bar_length = int(percent / 2)
        bar = "█" * bar_length
        print(f"   {level:10s} {bar} {percent:.1f}%")
    
    # Configuration
    print("\n⚙️  Configuration:")
    config = report['monitoring_config']
    print(f"   Device: {config['device']}")
    print(f"   Memory limit: {config['memory_limit_gb']:.2f}GB")
    print(f"   Sampling interval: {config['sampling_interval_ms']:.0f}ms")
    print(f"   Alerts enabled: {config['alerts_enabled']}")
    
    # Cleanup
    monitor.cleanup()


def demo_context_manager():
    """Demonstrate context manager usage."""
    print("\n" + "=" * 80)
    print("Demo 4: Context Manager Usage")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n📊 Using MemoryMonitor as context manager...")
    
    # Monitor automatically starts and stops
    with MemoryMonitor(
        device=device,
        memory_limit_gb=2.0,
        sampling_interval_ms=100.0,
        enable_alerts=True,
        alert_callback=alert_callback
    ) as monitor:
        print("   ✓ Monitoring started automatically")
        
        # Do work
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)
            time.sleep(0.2)
        
        # Get snapshot
        snapshot = monitor.get_current_snapshot()
        print(f"   Current usage: {snapshot.allocated_gb:.3f}GB")
        
        # Cleanup tensors
        del tensors
    
    print("   ✓ Monitoring stopped automatically")


def demo_oom_detection():
    """Demonstrate OOM event detection."""
    print("\n" + "=" * 80)
    print("Demo 5: OOM Event Detection")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    monitor = MemoryMonitor(
        device=device,
        memory_limit_gb=2.0,
        enable_alerts=True,
        alert_callback=alert_callback
    )
    
    print("\n📊 Simulating OOM event...")
    
    # Record OOM event
    monitor.record_oom_event()
    
    print(f"   OOM events recorded: {monitor.oom_events}")
    
    # Check alerts
    recent_alerts = monitor.get_recent_alerts(count=1)
    if recent_alerts:
        alert = recent_alerts[0]
        print(f"   Alert generated: {alert.message}")
        print(f"   Severity: {alert.severity}")
        print(f"   Type: {alert.alert_type}")
    
    monitor.cleanup()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Memory Monitoring and Alerting System Demo")
    print("=" * 80)
    
    try:
        # Run demos
        demo_basic_monitoring()
        demo_pressure_detection()
        demo_analytics_reporting()
        demo_context_manager()
        demo_oom_detection()
        
        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
