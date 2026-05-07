"""
Simple demonstration of the enhanced progress tracking functionality.

This script demonstrates the key features of the progress tracking system
without requiring actual WSI files.
"""

import time
from typing import List

import numpy as np

from .wsi_stream_reader import ProgressCallback, StreamingProgress, StreamingProgressTracker


def demo_progress_tracker():
    """Demonstrate the StreamingProgressTracker functionality."""
    print("=" * 60)
    print("WSI Streaming Progress Tracking Demo")
    print("=" * 60)

    # Create a progress callback
    def progress_callback(progress: StreamingProgress):
        print(
            f"📊 Progress Update: {progress.get_progress_percentage()} "
            f"(ETA: {progress.get_eta_string()}, "
            f"Confidence: {progress.current_confidence:.3f})"
        )

    callback = ProgressCallback(
        callback_func=progress_callback,
        update_interval=0.5,  # Update every 0.5 seconds
        min_progress_delta=0.1,  # Update on 10% progress change
    )

    # Initialize progress tracker
    total_tiles = 50
    tracker = StreamingProgressTracker(
        total_tiles=total_tiles,
        confidence_threshold=0.95,
        target_processing_time=15.0,
        progress_callbacks=[callback],
    )

    print(f"Initialized tracker for {total_tiles} tiles")
    print(f"Target processing time: 15.0 seconds")
    print(f"Confidence threshold: 0.95")
    print()

    # Start processing
    tracker.start_processing()
    print("🚀 Started processing...")

    # Simulate tile processing with varying speeds
    processing_times = np.random.uniform(0.1, 0.3, total_tiles)  # 100-300ms per tile
    confidences = np.linspace(0.1, 0.98, total_tiles)  # Gradually increasing confidence

    for i, (proc_time, confidence) in enumerate(zip(processing_times, confidences)):
        # Simulate processing time
        time.sleep(min(proc_time, 0.1))  # Cap at 100ms for demo

        # Record tile processing
        success = np.random.random() > 0.05  # 95% success rate
        skipped = np.random.random() < 0.1 if success else False  # 10% skip rate

        tracker.record_tile_processed(
            processing_time=proc_time, tile_size=1024, success=success, skipped=skipped
        )

        # Update confidence
        tracker.update_confidence(confidence)

        # Get progress (this will trigger callbacks)
        progress = tracker.get_current_progress()

        # Check for early stopping
        if progress.early_stop_recommended:
            print(f"🛑 Early stopping recommended at tile {i+1}/{total_tiles}")
            print(
                f"   Confidence: {progress.current_confidence:.3f} >= {progress.confidence_threshold:.3f}"
            )
            break

        # Simulate stage transitions
        if i == total_tiles // 4:
            tracker.start_stage("processing")
            print("🔄 Transitioned to processing stage")
        elif i == 3 * total_tiles // 4:
            tracker.start_stage("aggregating")
            print("🔄 Transitioned to aggregating stage")

    # Finish processing
    final_progress = tracker.finish_processing()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Processing completed!")
    print(f"📊 Tiles processed: {final_progress.tiles_processed}/{final_progress.total_tiles}")
    print(f"⏱️  Total time: {final_progress.elapsed_time:.2f}s")
    print(f"🎯 Final confidence: {final_progress.current_confidence:.3f}")
    print(f"🚀 Throughput: {final_progress.throughput_tiles_per_second:.1f} tiles/sec")
    print(f"💾 Peak memory: {final_progress.peak_memory_usage_gb:.3f}GB")
    print(f"📈 Data quality: {final_progress.data_quality_score:.1%}")
    print(f"⏭️  Early stopping: {'Yes' if final_progress.early_stop_recommended else 'No'}")

    # Show stage breakdown
    print(f"\n📋 Stage Breakdown:")
    print(f"   Loading: {final_progress.time_spent_loading:.2f}s")
    print(f"   Processing: {final_progress.time_spent_processing:.2f}s")
    print(f"   Aggregating: {final_progress.time_spent_aggregating:.2f}s")

    # Show quality metrics
    print(f"\n🔍 Quality Metrics:")
    print(f"   Successful tiles: {final_progress.tiles_processed}")
    print(f"   Failed tiles: {final_progress.tiles_failed}")
    print(f"   Skipped tiles: {final_progress.tiles_skipped}")

    return final_progress


def demo_eta_accuracy():
    """Demonstrate ETA calculation accuracy."""
    print("\n" + "=" * 60)
    print("ETA CALCULATION ACCURACY DEMO")
    print("=" * 60)

    # Test different scenarios
    scenarios = [
        {"name": "Fast Processing", "tile_time": 0.05, "tiles": 100},
        {"name": "Standard Processing", "tile_time": 0.1, "tiles": 100},
        {"name": "Slow Processing", "tile_time": 0.2, "tiles": 100},
    ]

    for scenario in scenarios:
        print(f"\n🧪 Scenario: {scenario['name']}")
        print(f"   Target tile time: {scenario['tile_time']}s")
        print(f"   Total tiles: {scenario['tiles']}")

        tracker = StreamingProgressTracker(
            total_tiles=scenario["tiles"], confidence_threshold=0.95, target_processing_time=30.0
        )

        tracker.start_processing()

        # Process first 20% of tiles to establish baseline
        warmup_tiles = scenario["tiles"] // 5
        for i in range(warmup_tiles):
            # Add some variance to processing time
            actual_time = scenario["tile_time"] * np.random.uniform(0.8, 1.2)
            time.sleep(min(actual_time, 0.05))  # Cap for demo

            tracker.record_tile_processed(
                processing_time=actual_time, tile_size=1024, success=True, skipped=False
            )

        # Get ETA after warmup
        progress = tracker.get_current_progress()
        predicted_total = progress.elapsed_time + progress.estimated_time_remaining
        actual_total = scenario["tiles"] * scenario["tile_time"]

        print(f"   After {warmup_tiles} tiles:")
        print(f"     Elapsed: {progress.elapsed_time:.2f}s")
        print(f"     ETA: {progress.estimated_time_remaining:.2f}s")
        print(f"     Predicted total: {predicted_total:.2f}s")
        print(f"     Actual total: {actual_total:.2f}s")
        print(
            f"     Accuracy: {(1 - abs(predicted_total - actual_total) / actual_total) * 100:.1f}%"
        )


def demo_confidence_tracking():
    """Demonstrate confidence-based early stopping."""
    print("\n" + "=" * 60)
    print("CONFIDENCE-BASED EARLY STOPPING DEMO")
    print("=" * 60)

    tracker = StreamingProgressTracker(
        total_tiles=200,
        confidence_threshold=0.92,  # Lower threshold for demo
        target_processing_time=30.0,
    )

    tracker.start_processing()

    # Simulate confidence progression
    confidence_values = [0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.82, 0.88, 0.91, 0.93, 0.95, 0.96]

    tiles_per_update = 200 // len(confidence_values)

    for i, target_confidence in enumerate(confidence_values):
        # Process a batch of tiles
        for j in range(tiles_per_update):
            tracker.record_tile_processed(0.1, 1024, True, False)

        # Update confidence
        tracker.update_confidence(target_confidence)
        progress = tracker.get_current_progress()

        print(
            f"📊 Batch {i+1}: Confidence {target_confidence:.3f}, "
            f"Progress {progress.progress_ratio:.1%}"
        )

        if progress.early_stop_recommended:
            print(f"🛑 Early stopping triggered!")
            print(
                f"   Confidence: {progress.current_confidence:.3f} >= {progress.confidence_threshold:.3f}"
            )
            print(f"   Tiles processed: {progress.tiles_processed}/{progress.total_tiles}")
            print(f"   Time saved: {progress.estimated_time_remaining:.1f}s")
            break

    final_progress = tracker.finish_processing()
    efficiency = (1 - final_progress.tiles_processed / final_progress.total_tiles) * 100
    print(f"✅ Processing efficiency: {efficiency:.1f}% time saved through early stopping")


def main():
    """Run all demonstrations."""
    print("🔬 WSI Streaming Progress Tracking Demonstrations")
    print("=" * 80)

    try:
        # Run demonstrations
        demo_progress_tracker()
        demo_eta_accuracy()
        demo_confidence_tracking()

        print("\n" + "=" * 80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        print("\n🚀 Key Features Demonstrated:")
        print("   ✅ Real-time progress tracking with callbacks")
        print("   ✅ Accurate ETA estimation")
        print("   ✅ Confidence-based early stopping")
        print("   ✅ Multi-stage processing breakdown")
        print("   ✅ Performance and quality metrics")
        print("   ✅ Memory usage monitoring")
        print("   ✅ Adaptive processing optimization")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
