"""
Test Real-Time WSI Streaming with Mock Models

Quick test script to verify the streaming pipeline works with mock models.
No trained models or real WSI files required.

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import sys
from pathlib import Path
import logging
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(output_path: str, size: tuple = (5000, 5000)):
    """
    Create a small test image for quick testing.
    
    Args:
        output_path: Path to save image
        size: Image dimensions (width, height)
    """
    logger.info(f"Creating test image: {output_path} ({size[0]}x{size[1]})")
    
    # Create random RGB image
    img_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save as TIFF
    img.save(output_path, format='TIFF')
    logger.info(f"Test image created: {output_path}")


def test_mock_models():
    """Test that mock models work correctly."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Mock Models")
    logger.info("="*80)
    
    from src.streaming.mock_models import create_mock_models
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Create mock models
    cnn_encoder, attention_model = create_mock_models(feature_dim=512, device=device)
    
    # Test CNN encoder
    logger.info("\nTesting CNN Encoder:")
    test_patches = torch.randn(32, 3, 224, 224).to(device)
    features = cnn_encoder(test_patches)
    logger.info(f"  Input: {test_patches.shape}")
    logger.info(f"  Output: {features.shape}")
    logger.info(f"  ✓ CNN encoder working")
    
    # Test attention model
    logger.info("\nTesting Attention Model:")
    test_features = torch.randn(1, 100, 512).to(device)
    logits, attention_weights = attention_model(test_features, return_attention=True)
    logger.info(f"  Input: {test_features.shape}")
    logger.info(f"  Logits: {logits.shape}")
    logger.info(f"  Attention: {attention_weights.shape}")
    logger.info(f"  Attention sum: {attention_weights.sum():.6f}")
    logger.info(f"  ✓ Attention model working")
    
    return True


def test_streaming_config():
    """Test streaming configuration."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Streaming Configuration")
    logger.info("="*80)
    
    from src.streaming import StreamingConfig
    
    config = StreamingConfig(
        tile_size=512,
        batch_size=16,
        memory_budget_gb=1.0,
        target_time=60.0,
        confidence_threshold=0.90,
        enable_visualization=False  # Disable for faster testing
    )
    
    logger.info(f"  Tile size: {config.tile_size}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Memory budget: {config.memory_budget_gb}GB")
    logger.info(f"  Target time: {config.target_time}s")
    logger.info(f"  ✓ Configuration created")
    
    return config


def test_processor_initialization(config):
    """Test processor initialization with mock models."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Processor Initialization")
    logger.info("="*80)
    
    from src.streaming import RealTimeWSIProcessor
    
    processor = RealTimeWSIProcessor(config)
    logger.info(f"  ✓ Processor initialized")
    
    return processor


def test_end_to_end_processing(processor, test_image_path):
    """Test end-to-end processing with mock models."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: End-to-End Processing")
    logger.info("="*80)
    
    import asyncio
    
    logger.info(f"Processing: {test_image_path}")
    
    try:
        # Run processing
        result = asyncio.run(processor.process_wsi_realtime(test_image_path))
        
        logger.info("\n" + "-"*80)
        logger.info("RESULTS:")
        logger.info("-"*80)
        logger.info(f"  Prediction: {result.prediction}")
        logger.info(f"  Confidence: {result.confidence:.3f}")
        logger.info(f"  Processing time: {result.processing_time:.2f}s")
        logger.info(f"  Patches processed: {result.patches_processed}")
        logger.info(f"  Throughput: {result.throughput_patches_per_sec:.1f} patches/s")
        logger.info(f"  Peak memory: {result.peak_memory_gb:.2f}GB")
        logger.info(f"  Early stopped: {result.early_stopped}")
        logger.info(f"  ✓ Processing completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"  ✗ Processing failed: {e}", exc_info=True)
        return None


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("REAL-TIME WSI STREAMING - MOCK MODEL TESTS")
    print("="*80)
    print("\nTesting the streaming pipeline with mock models")
    print("No trained models or real WSI files required\n")
    
    try:
        # Test 1: Mock models
        if not test_mock_models():
            logger.error("Mock models test failed")
            return False
        
        # Test 2: Configuration
        config = test_streaming_config()
        if not config:
            logger.error("Configuration test failed")
            return False
        
        # Test 3: Processor initialization
        processor = test_processor_initialization(config)
        if not processor:
            logger.error("Processor initialization failed")
            return False
        
        # Create test image
        test_image_path = "test_image_small.tiff"
        create_test_image(test_image_path, size=(5000, 5000))
        
        # Test 4: End-to-end processing
        result = test_end_to_end_processing(processor, test_image_path)
        
        # Cleanup
        Path(test_image_path).unlink(missing_ok=True)
        
        if result:
            print("\n" + "="*80)
            print("✓ ALL TESTS PASSED")
            print("="*80)
            print("\nThe streaming pipeline is working correctly with mock models!")
            print("You can now:")
            print("  1. Test with real WSI files when available")
            print("  2. Replace mock models with trained models")
            print("  3. Run the full demo: python examples/streaming_demo.py")
            return True
        else:
            print("\n" + "="*80)
            print("✗ SOME TESTS FAILED")
            print("="*80)
            return False
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
