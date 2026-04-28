#!/usr/bin/env python3
"""
Production System Test Script

Tests the production Medical AI system with real database and model inference.
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image() -> bytes:
    """Create a synthetic test image."""
    # Create a 224x224 RGB image with some pattern
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some structure to make it look more like tissue
    for i in range(0, 224, 20):
        for j in range(0, 224, 20):
            center_x, center_y = j + 10, i + 10
            for y in range(max(0, i), min(224, i + 20)):
                for x in range(max(0, j), min(224, j + 20)):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < 8:
                        image[y, x] = [200, 150, 200]  # Purple-ish
    
    # Convert to bytes
    pil_image = Image.fromarray(image)
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def test_production_system(base_url: str = "http://localhost:8000"):
    """Test the production system."""
    
    logger.info(f"Testing production system at {base_url}")
    
    # Test 1: Health check
    logger.info("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ Health check passed: {health_data['status']}")
            
            # Check components
            components = health_data.get('components', {})
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"   {status_icon} {component}: {'OK' if status else 'FAILED'}")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Database connectivity
    logger.info("2. Testing database connectivity...")
    try:
        response = requests.get(f"{base_url}/api/v1/system/db-health", timeout=10)
        if response.status_code == 200:
            db_health = response.json()
            logger.info(f"✅ Database health: {db_health['status']}")
            if 'query_time_ms' in db_health:
                logger.info(f"   Query time: {db_health['query_time_ms']:.2f}ms")
        else:
            logger.error(f"❌ Database health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Database health check error: {e}")
        return False
    
    # Test 3: Real model inference
    logger.info("3. Testing real model inference...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Upload for analysis
        files = {'file': ('test_image.png', test_image, 'image/png')}
        response = requests.post(
            f"{base_url}/api/v1/analyze/upload",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            upload_result = response.json()
            analysis_id = upload_result.get('analysis_id')
            logger.info(f"✅ Image uploaded: {analysis_id}")
            
            # Poll for results
            max_wait = 60
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                response = requests.get(
                    f"{base_url}/api/v1/analyze/{analysis_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get('status')
                    
                    if status == 'completed':
                        logger.info(f"✅ Analysis completed!")
                        logger.info(f"   Prediction: {result.get('prediction_class')}")
                        logger.info(f"   Confidence: {result.get('confidence_score', 0):.3f}")
                        logger.info(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
                        logger.info(f"   Model: {result.get('model_name', 'unknown')}")
                        
                        # Validate result structure
                        required_fields = ['prediction_class', 'confidence_score', 'processing_time_ms']
                        missing_fields = [f for f in required_fields if f not in result]
                        
                        if missing_fields:
                            logger.warning(f"Missing fields in result: {missing_fields}")
                        
                        break
                    elif status == 'failed':
                        logger.error(f"❌ Analysis failed: {result}")
                        return False
                    else:
                        logger.info(f"   Status: {status}")
                        time.sleep(2)
                else:
                    logger.error(f"❌ Failed to get analysis status: {response.status_code}")
                    return False
            else:
                logger.error("❌ Analysis timeout")
                return False
                
        else:
            logger.error(f"❌ Image upload failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model inference test error: {e}")
        return False
    
    # Test 4: Database operations
    logger.info("4. Testing database operations...")
    try:
        # Create a test case
        case_data = {
            "patient_id": f"TEST_PAT_{int(time.time())}",
            "study_id": f"TEST_STU_{int(time.time())}",
            "case_type": "breast_cancer_screening",
            "priority": "normal"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/cases",
            json=case_data,
            timeout=10
        )
        
        if response.status_code == 200:
            case_result = response.json()
            case_id = case_result.get('case_id')
            logger.info(f"✅ Case created: {case_id}")
            
            # Retrieve the case
            response = requests.get(f"{base_url}/api/v1/cases/{case_id}", timeout=10)
            if response.status_code == 200:
                case_details = response.json()
                logger.info(f"✅ Case retrieved: {case_details['patient_id']}")
            else:
                logger.error(f"❌ Failed to retrieve case: {response.status_code}")
                return False
                
        else:
            logger.error(f"❌ Case creation failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database operations test error: {e}")
        return False
    
    # Test 5: Analytics endpoints
    logger.info("5. Testing analytics endpoints...")
    try:
        response = requests.get(f"{base_url}/api/v1/analytics/dashboard", timeout=10)
        if response.status_code == 200:
            dashboard_data = response.json()
            logger.info(f"✅ Dashboard data retrieved")
            logger.info(f"   Total cases: {dashboard_data.get('total_cases', 0)}")
            logger.info(f"   Completed analyses: {dashboard_data.get('completed_analyses', 0)}")
        else:
            logger.error(f"❌ Dashboard data failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Analytics test error: {e}")
        return False
    
    logger.info("🎉 All production system tests passed!")
    return True


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test production Medical AI system")
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='Base URL for API server'
    )
    
    args = parser.parse_args()
    
    success = test_production_system(args.base_url)
    
    if success:
        logger.info("✅ Production system is working correctly!")
        sys.exit(0)
    else:
        logger.error("❌ Production system tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()