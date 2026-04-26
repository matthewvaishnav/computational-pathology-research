"""Demo script for web dashboard with simulated WSI processing.

This script demonstrates how to integrate the web dashboard with a WSI processing pipeline.
It simulates real-time processing updates for demonstration purposes.
"""

import asyncio
import numpy as np
import logging
from pathlib import Path
import time

from src.streaming.web_dashboard import (
    update_dashboard_status,
    update_dashboard_error,
    update_dashboard_complete,
    dashboard_state
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_wsi_processing(
    slide_id: str = "demo_slide",
    total_patches: int = 1000,
    batch_size: int = 64,
    heatmap_size: tuple = (50, 50),
    target_confidence: float = 0.95,
    processing_time: float = 30.0
):
    """Simulate WSI processing with dashboard updates.
    
    Args:
        slide_id: Identifier for the slide
        total_patches: Total number of patches to process
        batch_size: Patches per batch
        heatmap_size: Dimensions of attention heatmap (width, height)
        target_confidence: Target confidence threshold
        processing_time: Total time to simulate (seconds)
    """
    logger.info(f"Starting simulated processing for {slide_id}")
    logger.info(f"Total patches: {total_patches}, Batch size: {batch_size}")
    
    # Initialize dashboard state
    dashboard_state.reset()
    dashboard_state.slide_id = slide_id
    dashboard_state.status = "processing"
    dashboard_state.total_patches = total_patches
    dashboard_state.start_time = time.time()
    dashboard_state.heatmap_dimensions = heatmap_size
    dashboard_state.attention_heatmap = np.zeros(heatmap_size, dtype=np.float32)
    dashboard_state.coverage_mask = np.zeros(heatmap_size, dtype=bool)
    
    # Calculate timing
    num_batches = (total_patches + batch_size - 1) // batch_size
    delay_per_batch = processing_time / num_batches
    
    try:
        # Process batches
        for batch_idx in range(num_batches):
            # Calculate batch info
            patches_in_batch = min(batch_size, total_patches - batch_idx * batch_size)
            patches_processed = batch_idx * batch_size + patches_in_batch
            
            # Simulate confidence building (starts low, increases over time)
            progress = patches_processed / total_patches
            confidence = 0.5 + (progress * 0.5)  # 0.5 → 1.0
            
            # Add some realistic variation
            confidence += np.random.normal(0, 0.02)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Generate simulated attention weights and coordinates
            attention_weights = np.random.beta(2, 5, size=patches_in_batch).astype(np.float32)
            
            # Generate coordinates that cover the heatmap progressively
            coords = []
            for i in range(patches_in_batch):
                patch_idx = batch_idx * batch_size + i
                x = (patch_idx % heatmap_size[0])
                y = (patch_idx // heatmap_size[0]) % heatmap_size[1]
                coords.append([x, y])
            coordinates = np.array(coords)
            
            # Update dashboard
            await update_dashboard_status(
                patches_processed=patches_processed,
                total_patches=total_patches,
                confidence=confidence,
                attention_weights=attention_weights,
                coordinates=coordinates
            )
            
            # Log progress
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches}: "
                    f"{patches_processed}/{total_patches} patches, "
                    f"confidence={confidence:.3f}"
                )
            
            # Check for early stopping
            if confidence >= target_confidence and progress > 0.5:
                logger.info(f"Early stopping: confidence {confidence:.3f} >= {target_confidence}")
                break
            
            # Simulate processing time
            await asyncio.sleep(delay_per_batch)
        
        # Mark processing as complete
        await update_dashboard_complete()
        logger.info(f"Processing complete! Final confidence: {dashboard_state.current_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        await update_dashboard_error(str(e))
        raise


async def run_multiple_slides_demo():
    """Demo processing multiple slides sequentially."""
    slides = [
        ("slide_001", 800, 32),
        ("slide_002", 1200, 64),
        ("slide_003", 600, 48),
    ]
    
    for slide_id, total_patches, batch_size in slides:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {slide_id}")
        logger.info(f"{'='*60}\n")
        
        await simulate_wsi_processing(
            slide_id=slide_id,
            total_patches=total_patches,
            batch_size=batch_size,
            processing_time=20.0
        )
        
        # Wait between slides
        logger.info(f"\nWaiting 5 seconds before next slide...\n")
        await asyncio.sleep(5)


async def run_error_demo():
    """Demo error handling in dashboard."""
    logger.info("Starting error demo...")
    
    dashboard_state.reset()
    dashboard_state.slide_id = "error_slide"
    dashboard_state.status = "processing"
    dashboard_state.total_patches = 1000
    dashboard_state.start_time = time.time()
    
    try:
        # Process a few batches
        for i in range(5):
            await update_dashboard_status(
                patches_processed=i * 100,
                total_patches=1000,
                confidence=0.5 + i * 0.05
            )
            await asyncio.sleep(1)
        
        # Simulate error
        raise RuntimeError("Simulated processing error for demo")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        await update_dashboard_error(str(e))


async def run_parameter_update_demo():
    """Demo updating parameters during processing."""
    logger.info("Starting parameter update demo...")
    
    # Start processing
    dashboard_state.reset()
    dashboard_state.slide_id = "param_demo_slide"
    dashboard_state.status = "processing"
    dashboard_state.total_patches = 1000
    dashboard_state.start_time = time.time()
    
    # Process with initial parameters
    logger.info(f"Initial parameters: {dashboard_state.parameters}")
    
    for i in range(10):
        await update_dashboard_status(
            patches_processed=i * 100,
            total_patches=1000,
            confidence=0.5 + i * 0.04
        )
        
        # Simulate parameter update mid-processing
        if i == 5:
            logger.info("Updating parameters mid-processing...")
            dashboard_state.parameters.confidence_threshold = 0.90
            dashboard_state.parameters.batch_size = 128
            logger.info(f"New parameters: {dashboard_state.parameters}")
        
        await asyncio.sleep(1)
    
    await update_dashboard_complete()


def main():
    """Main entry point for demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Dashboard Demo")
    parser.add_argument(
        "--mode",
        choices=["single", "multiple", "error", "parameters"],
        default="single",
        help="Demo mode to run"
    )
    parser.add_argument(
        "--patches",
        type=int,
        default=1000,
        help="Total patches to process (single mode)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (single mode)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=30.0,
        help="Processing time in seconds (single mode)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Web Dashboard Demo")
    logger.info("="*60)
    logger.info("\nMake sure the dashboard server is running:")
    logger.info("  uvicorn src.streaming.web_dashboard:app --reload")
    logger.info("\nThen open: http://localhost:8000")
    logger.info("="*60 + "\n")
    
    # Run selected demo
    if args.mode == "single":
        asyncio.run(simulate_wsi_processing(
            total_patches=args.patches,
            batch_size=args.batch_size,
            processing_time=args.time
        ))
    elif args.mode == "multiple":
        asyncio.run(run_multiple_slides_demo())
    elif args.mode == "error":
        asyncio.run(run_error_demo())
    elif args.mode == "parameters":
        asyncio.run(run_parameter_update_demo())


if __name__ == "__main__":
    main()
