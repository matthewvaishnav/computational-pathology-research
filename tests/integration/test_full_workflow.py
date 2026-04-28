#!/usr/bin/env python3
"""
Integration Tests - Full Workflow Testing

Tests the complete Medical AI platform workflow from image upload to results.
Validates API endpoints, database operations, model inference, and DICOM integration.
"""

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import requests
from PIL import Image
from pydicom import Dataset, dcmwrite
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class IntegrationTestSuite:
    """Comprehensive integration test suite for Medical AI platform."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize test suite with API base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        
        # Test configuration
        self.test_config = {
            'timeout': 30,
            'max_retries': 3,
            'test_image_size': (224, 224),
            'confidence_threshold': 0.5
        }
        
        print(f"🧪 Integration Test Suite initialized")
        print(f"📡 API Base URL: {self.base_url}")
    
    def wait_for_service(self, endpoint: str, timeout: int = 60) -> bool:
        """Wait for service to be ready."""
        print(f"⏳ Waiting for service: {endpoint}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Service ready: {endpoint}")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        print(f"❌ Service not ready: {endpoint}")
        return False
    
    def create_test_image(self, width: int = 224, height: int = 224) -> bytes:
        """Create a synthetic test image for testing."""
        # Create a synthetic pathology-like image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Simulate tissue patterns
        for i in range(0, height, 20):
            for j in range(0, width, 20):
                # Add circular structures (simulate cells)
                center_x, center_y = j + 10, i + 10
                for y in range(max(0, i), min(height, i + 20)):
                    for x in range(max(0, j), min(width, j + 20)):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < 8:
                            image[y, x] = [200, 150, 200]  # Purple-ish (H&E staining)
        
        # Convert to PIL Image and save to bytes
        pil_image = Image.fromarray(image)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def create_test_dicom(self) -> bytes:
        """Create a synthetic DICOM file for testing."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import tempfile
        
        # Create file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"  # VL Whole Slide Microscopy
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian  # Explicit VR Little Endian
        file_meta.ImplementationClassUID = generate_uid()
        
        # Create the FileDataset instance
        ds = FileDataset(
            filename="test.dcm",
            dataset={},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )
        
        # Required DICOM tags
        ds.PatientName = "TEST^PATIENT"
        ds.PatientID = "TEST001"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "SM"
        ds.StudyDate = "20260427"
        ds.StudyTime = "120000"
        ds.SeriesNumber = 1
        ds.InstanceNumber = 1
        
        # Image data - create simple RGB image
        test_image = self.create_test_image(512, 512)
        ds.PixelData = test_image
        ds.Rows = 512
        ds.Columns = 512
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0
        
        # Save to bytes
        dicom_bytes = io.BytesIO()
        ds.save_as(dicom_bytes, write_like_original=False)
        dicom_bytes.seek(0)
        
        return dicom_bytes.getvalue()
    
    def test_health_check(self) -> bool:
        """Test API health check endpoint."""
        print("\n🔍 Testing Health Check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_api_documentation(self) -> bool:
        """Test API documentation endpoint."""
        print("\n📚 Testing API Documentation...")
        
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=10)
            
            if response.status_code == 200:
                print("✅ API documentation accessible")
                return True
            else:
                print(f"❌ API documentation failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ API documentation error: {e}")
            return False
    
    def test_image_upload_and_analysis(self) -> bool:
        """Test complete image upload and analysis workflow."""
        print("\n🖼️ Testing Image Upload and Analysis...")
        
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Upload image for analysis
            files = {'file': ('test_image.png', test_image, 'image/png')}
            response = self.session.post(
                f"{self.base_url}/api/v1/analyze/upload",
                files=files,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Image upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            upload_result = response.json()
            analysis_id = upload_result.get('analysis_id')
            
            if not analysis_id:
                print("❌ No analysis ID returned")
                return False
            
            print(f"✅ Image uploaded successfully: {analysis_id}")
            
            # Poll for analysis results
            max_wait = 60  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                response = self.session.get(
                    f"{self.base_url}/api/v1/analyze/{analysis_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get('status')
                    
                    if status == 'completed':
                        print(f"✅ Analysis completed: {result}")
                        
                        # Validate result structure
                        required_fields = ['confidence_score', 'prediction_class', 'processing_time_ms']
                        for field in required_fields:
                            if field not in result:
                                print(f"❌ Missing field in result: {field}")
                                return False
                        
                        # Validate confidence score
                        confidence = result.get('confidence_score', 0)
                        if not (0 <= confidence <= 1):
                            print(f"❌ Invalid confidence score: {confidence}")
                            return False
                        
                        print("✅ Analysis result validation passed")
                        return True
                    
                    elif status == 'failed':
                        print(f"❌ Analysis failed: {result}")
                        return False
                    
                    else:
                        print(f"⏳ Analysis in progress: {status}")
                        time.sleep(2)
                
                else:
                    print(f"❌ Failed to get analysis status: {response.status_code}")
                    return False
            
            print("❌ Analysis timeout")
            return False
            
        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            return False
    
    def test_dicom_integration(self) -> bool:
        """Test DICOM integration workflow."""
        print("\n🏥 Testing DICOM Integration...")
        
        try:
            # Create test DICOM file
            dicom_data = self.create_test_dicom()
            
            # Upload DICOM file
            files = {'file': ('test_study.dcm', dicom_data, 'application/dicom')}
            response = self.session.post(
                f"{self.base_url}/api/v1/dicom/upload",
                files=files,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ DICOM upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            upload_result = response.json()
            study_id = upload_result.get('study_id')
            
            if not study_id:
                print("❌ No study ID returned")
                return False
            
            print(f"✅ DICOM uploaded successfully: {study_id}")
            
            # Query DICOM study
            response = self.session.get(
                f"{self.base_url}/api/v1/dicom/study/{study_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                study_data = response.json()
                print(f"✅ DICOM study retrieved: {study_data}")
                
                # Validate study data
                required_fields = ['study_instance_uid', 'patient_id', 'study_date']
                for field in required_fields:
                    if field not in study_data:
                        print(f"❌ Missing field in study data: {field}")
                        return False
                
                return True
            else:
                print(f"❌ Failed to retrieve DICOM study: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ DICOM integration error: {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test system performance benchmarks."""
        print("\n⚡ Testing Performance Benchmarks...")
        
        try:
            # Test multiple concurrent requests
            num_requests = 5
            test_images = [self.create_test_image() for _ in range(num_requests)]
            
            start_time = time.time()
            analysis_ids = []
            
            # Submit all requests
            for i, test_image in enumerate(test_images):
                files = {'file': (f'test_image_{i}.png', test_image, 'image/png')}
                response = self.session.post(
                    f"{self.base_url}/api/v1/analyze/upload",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis_ids.append(result.get('analysis_id'))
                else:
                    print(f"❌ Request {i} failed: {response.status_code}")
                    return False
            
            upload_time = time.time() - start_time
            print(f"📊 Upload time for {num_requests} requests: {upload_time:.2f}s")
            
            # Wait for all analyses to complete
            completed_count = 0
            max_wait = 120  # seconds
            start_wait = time.time()
            
            while completed_count < num_requests and time.time() - start_wait < max_wait:
                for analysis_id in analysis_ids:
                    if analysis_id:
                        response = self.session.get(
                            f"{self.base_url}/api/v1/analyze/{analysis_id}",
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('status') == 'completed':
                                completed_count += 1
                
                time.sleep(1)
            
            total_time = time.time() - start_time
            
            if completed_count == num_requests:
                print(f"✅ All {num_requests} analyses completed in {total_time:.2f}s")
                print(f"📊 Average time per analysis: {total_time/num_requests:.2f}s")
                
                # Performance thresholds
                if total_time / num_requests > 60:  # 60 seconds per analysis
                    print("⚠️ Performance warning: Analysis time exceeds threshold")
                
                return True
            else:
                print(f"❌ Only {completed_count}/{num_requests} analyses completed")
                return False
                
        except Exception as e:
            print(f"❌ Performance benchmark error: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and edge cases."""
        print("\n🚨 Testing Error Handling...")
        
        try:
            # Test invalid file upload
            invalid_data = b"This is not an image"
            files = {'file': ('invalid.txt', invalid_data, 'text/plain')}
            response = self.session.post(
                f"{self.base_url}/api/v1/analyze/upload",
                files=files,
                timeout=10
            )
            
            if response.status_code == 400:
                print("✅ Invalid file upload properly rejected")
            else:
                print(f"❌ Invalid file upload not handled: {response.status_code}")
                return False
            
            # Test missing file upload
            response = self.session.post(
                f"{self.base_url}/api/v1/analyze/upload",
                timeout=10
            )
            
            if response.status_code == 422:
                print("✅ Missing file upload properly rejected")
            else:
                print(f"❌ Missing file upload not handled: {response.status_code}")
                return False
            
            # Test invalid analysis ID
            response = self.session.get(
                f"{self.base_url}/api/v1/analyze/invalid-id",
                timeout=10
            )
            
            if response.status_code == 404:
                print("✅ Invalid analysis ID properly handled")
            else:
                print(f"❌ Invalid analysis ID not handled: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test error: {e}")
            return False
    
    def test_database_operations(self) -> bool:
        """Test database connectivity and operations."""
        print("\n🗄️ Testing Database Operations...")
        
        try:
            # Test database health through API
            response = self.session.get(f"{self.base_url}/api/v1/system/db-health", timeout=10)
            
            if response.status_code == 200:
                db_health = response.json()
                print(f"✅ Database health check passed: {db_health}")
                
                # Validate database metrics
                if 'connection_count' in db_health and 'query_time_ms' in db_health:
                    return True
                else:
                    print("❌ Missing database health metrics")
                    return False
            else:
                print(f"❌ Database health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Database operations error: {e}")
            return False
    
    def test_monitoring_endpoints(self) -> bool:
        """Test monitoring and metrics endpoints."""
        print("\n📊 Testing Monitoring Endpoints...")
        
        try:
            # Test metrics endpoint
            response = self.session.get(f"{self.base_url}/metrics", timeout=10)
            
            if response.status_code == 200:
                metrics_data = response.text
                
                # Check for key metrics
                required_metrics = [
                    'api_requests_total',
                    'api_request_duration_seconds',
                    'model_inference_duration_seconds'
                ]
                
                for metric in required_metrics:
                    if metric in metrics_data:
                        print(f"✅ Metric found: {metric}")
                    else:
                        print(f"❌ Missing metric: {metric}")
                        return False
                
                return True
            else:
                print(f"❌ Metrics endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Monitoring endpoints error: {e}")
            return False
    
    def test_ci_cd_integration(self) -> bool:
        """Test CI/CD specific endpoints and functionality."""
        print("\n🔄 Testing CI/CD Integration...")
        
        try:
            # Test build info endpoint
            response = self.session.get(f"{self.base_url}/api/v1/system/build-info", timeout=10)
            
            if response.status_code == 200:
                build_info = response.json()
                print(f"✅ Build info retrieved: {build_info.get('version', 'unknown')}")
                
                # Validate build info structure
                required_fields = ['version', 'commit_hash', 'build_date']
                for field in required_fields:
                    if field not in build_info:
                        print(f"⚠️ Missing build info field: {field}")
            else:
                print(f"❌ Build info failed: {response.status_code}")
                return False
            
            # Test deployment readiness
            response = self.session.get(f"{self.base_url}/api/v1/system/readiness", timeout=10)
            
            if response.status_code == 200:
                readiness = response.json()
                print(f"✅ Deployment readiness: {readiness.get('ready', False)}")
                
                # Check readiness components
                components = readiness.get('components', {})
                for component, status in components.items():
                    status_icon = "✅" if status else "❌"
                    print(f"   {status_icon} {component}: {'Ready' if status else 'Not Ready'}")
            else:
                print(f"❌ Readiness check failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ CI/CD integration test error: {e}")
            return False
    
    def test_security_headers(self) -> bool:
        """Test security headers and HTTPS configuration."""
        print("\n🔒 Testing Security Headers...")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                headers = response.headers
                
                # Check security headers
                security_headers = {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'DENY',
                    'X-XSS-Protection': '1; mode=block',
                    'Strict-Transport-Security': 'max-age=31536000',
                    'Content-Security-Policy': 'default-src'
                }
                
                missing_headers = []
                for header, expected in security_headers.items():
                    if header in headers:
                        print(f"✅ {header}: {headers[header]}")
                    else:
                        missing_headers.append(header)
                        print(f"❌ Missing security header: {header}")
                
                if missing_headers:
                    print(f"⚠️ {len(missing_headers)} security headers missing")
                    return False
                else:
                    print("✅ All security headers present")
                    return True
            else:
                print(f"❌ Security header check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Security header test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests and return results."""
        print("🚀 Starting Integration Test Suite")
        print("=" * 50)
        
        # Wait for services to be ready
        if not self.wait_for_service("/health"):
            print("❌ Services not ready, aborting tests")
            return {}
        
        # Define test methods
        tests = [
            ("Health Check", self.test_health_check),
            ("API Documentation", self.test_api_documentation),
            ("Image Upload & Analysis", self.test_image_upload_and_analysis),
            ("DICOM Integration", self.test_dicom_integration),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling", self.test_error_handling),
            ("Database Operations", self.test_database_operations),
            ("Monitoring Endpoints", self.test_monitoring_endpoints),
            ("CI/CD Integration", self.test_ci_cd_integration),
            ("Security Headers", self.test_security_headers)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        # Run each test
        for test_name, test_method in tests:
            try:
                print(f"\n{'='*20} {test_name} {'='*20}")
                result = test_method()
                results[test_name] = result
                
                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎯 INTEGRATION TEST SUMMARY")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<30} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Platform is ready for deployment.")
        else:
            print("⚠️ Some tests failed. Please review and fix issues before deployment.")
        
        return results


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Integration Test Suite")
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='Base URL for API server (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--wait-timeout',
        type=int,
        default=60,
        help='Timeout for waiting for services (default: 60 seconds)'
    )
    
    args = parser.parse_args()
    
    # Run integration tests
    test_suite = IntegrationTestSuite(base_url=args.base_url)
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()