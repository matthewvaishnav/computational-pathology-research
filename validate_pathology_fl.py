#!/usr/bin/env python3
"""PathologyFL validation and verification script."""

import time
import hashlib
from typing import Dict, List, Any

class PathologyFLValidator:
    """Validate PathologyFL operations and data integrity."""
    
    def __init__(self):
        self.validation_cache = {}
        
    def validate_hospital_metadata(self, metadata: Dict) -> Dict:
        """Validate hospital metadata completeness and format."""
        errors = []
        warnings = []
        
        required_fields = ["hospital_type", "years_experience", "diagnostic_accuracy"]
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
        
        # Validate ranges
        if "diagnostic_accuracy" in metadata:
            accuracy = metadata["diagnostic_accuracy"]
            if not (0.0 <= accuracy <= 1.0):
                errors.append(f"Diagnostic accuracy must be 0-1, got {accuracy}")
        
        if "years_experience" in metadata:
            years = metadata["years_experience"]
            if not (0 <= years <= 100):
                warnings.append(f"Unusual years_experience: {years}")
        
        return {"errors": errors, "warnings": warnings}
    
    def validate_model_parameters(self, parameters: Dict) -> Dict:
        """Validate model parameters for consistency."""
        errors = []
        warnings = []
        
        for layer_name, params in parameters.items():
            if not isinstance(params, list):
                errors.append(f"Layer {layer_name} parameters must be list")
                continue
                
            if len(params) == 0:
                warnings.append(f"Layer {layer_name} has no parameters")
                continue
            
            # Check for NaN or infinite values
            for i, param in enumerate(params):
                if not isinstance(param, (int, float)):
                    errors.append(f"Layer {layer_name}[{i}] must be numeric")
                elif param != param:  # NaN check
                    errors.append(f"Layer {layer_name}[{i}] is NaN")
                elif abs(param) > 1000:
                    warnings.append(f"Layer {layer_name}[{i}] has extreme value: {param}")
        
        return {"errors": errors, "warnings": warnings}
    
    def validate_aggregation_result(self, client_updates: List[Dict], result: Dict) -> Dict:
        """Validate aggregation result consistency."""
        errors = []
        warnings = []
        
        if not client_updates:
            errors.append("No client updates provided")
            return {"errors": errors, "warnings": warnings}
        
        # Check layer consistency
        expected_layers = set(client_updates[0]["parameters"].keys())
        result_layers = set(result.keys())
        
        if expected_layers != result_layers:
            errors.append(f"Layer mismatch: expected {expected_layers}, got {result_layers}")
        
        # Check parameter counts
        for layer in expected_layers & result_layers:
            expected_count = len(client_updates[0]["parameters"][layer])
            result_count = len(result[layer])
            
            if expected_count != result_count:
                errors.append(f"Layer {layer}: expected {expected_count} params, got {result_count}")
        
        return {"errors": errors, "warnings": warnings}
    
    def compute_checksum(self, data: Any) -> str:
        """Compute checksum for data integrity verification."""
        data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def validate_data_integrity(self, data: Any, expected_checksum: str) -> bool:
        """Validate data integrity using checksum."""
        actual_checksum = self.compute_checksum(data)
        return actual_checksum == expected_checksum

def test_metadata_validation():
    """Test hospital metadata validation."""
    print("Testing metadata validation...")
    
    validator = PathologyFLValidator()
    
    test_cases = [
        # Valid metadata
        ({
            "hospital_type": "cancer_center",
            "years_experience": 15,
            "diagnostic_accuracy": 0.92
        }, 0, 0),
        
        # Missing required field
        ({
            "hospital_type": "community",
            "years_experience": 10
        }, 1, 0),
        
        # Invalid accuracy range
        ({
            "hospital_type": "teaching_hospital",
            "years_experience": 20,
            "diagnostic_accuracy": 1.5
        }, 1, 0),
        
        # Warning case
        ({
            "hospital_type": "community",
            "years_experience": 150,  # Unusual
            "diagnostic_accuracy": 0.85
        }, 0, 1)
    ]
    
    passed = 0
    for metadata, expected_errors, expected_warnings in test_cases:
        result = validator.validate_hospital_metadata(metadata)
        
        if (len(result["errors"]) == expected_errors and 
            len(result["warnings"]) == expected_warnings):
            passed += 1
        else:
            print(f"  Failed validation: {result}")
    
    print(f"  Metadata validation: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_parameter_validation():
    """Test model parameter validation."""
    print("Testing parameter validation...")
    
    validator = PathologyFLValidator()
    
    test_cases = [
        # Valid parameters
        ({
            "layer1": [0.1, 0.2, 0.3],
            "layer2": [0.4, 0.5]
        }, 0, 0),
        
        # Empty layer
        ({
            "layer1": [],
            "layer2": [0.1, 0.2]
        }, 0, 1),
        
        # Non-numeric parameter
        ({
            "layer1": [0.1, "invalid", 0.3]
        }, 1, 0),
        
        # Extreme values
        ({
            "layer1": [0.1, 2000.0, 0.3]
        }, 0, 1)
    ]
    
    passed = 0
    for params, expected_errors, expected_warnings in test_cases:
        result = validator.validate_model_parameters(params)
        
        if (len(result["errors"]) == expected_errors and 
            len(result["warnings"]) == expected_warnings):
            passed += 1
    
    print(f"  Parameter validation: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_aggregation_validation():
    """Test aggregation result validation."""
    print("Testing aggregation validation...")
    
    validator = PathologyFLValidator()
    
    # Valid case
    client_updates = [
        {
            "hospital_id": "h1",
            "parameters": {
                "layer1": [0.1, 0.2],
                "layer2": [0.3, 0.4, 0.5]
            }
        },
        {
            "hospital_id": "h2", 
            "parameters": {
                "layer1": [0.2, 0.3],
                "layer2": [0.4, 0.5, 0.6]
            }
        }
    ]
    
    # Valid result
    valid_result = {
        "layer1": [0.15, 0.25],
        "layer2": [0.35, 0.45, 0.55]
    }
    
    # Invalid result (wrong parameter count)
    invalid_result = {
        "layer1": [0.15],  # Should have 2 parameters
        "layer2": [0.35, 0.45, 0.55]
    }
    
    valid_validation = validator.validate_aggregation_result(client_updates, valid_result)
    invalid_validation = validator.validate_aggregation_result(client_updates, invalid_result)
    
    valid_passed = len(valid_validation["errors"]) == 0
    invalid_failed = len(invalid_validation["errors"]) > 0
    
    print(f"  Valid result validation: {valid_passed}")
    print(f"  Invalid result detection: {invalid_failed}")
    
    return valid_passed and invalid_failed

def test_data_integrity():
    """Test data integrity verification."""
    print("Testing data integrity...")
    
    validator = PathologyFLValidator()
    
    # Test data
    test_data = {
        "hospital_id": "test_hospital",
        "parameters": {
            "layer1": [0.1, 0.2, 0.3],
            "layer2": [0.4, 0.5]
        }
    }
    
    # Compute checksum
    checksum = validator.compute_checksum(test_data)
    
    # Verify integrity
    integrity_valid = validator.validate_data_integrity(test_data, checksum)
    
    # Test with corrupted data
    corrupted_data = test_data.copy()
    corrupted_data["parameters"]["layer1"][0] = 999.0
    
    integrity_invalid = validator.validate_data_integrity(corrupted_data, checksum)
    
    print(f"  Original data checksum: {checksum[:8]}...")
    print(f"  Integrity validation: {integrity_valid}")
    print(f"  Corruption detection: {not integrity_invalid}")
    
    return integrity_valid and not integrity_invalid

def test_performance_validation():
    """Test validation performance with large datasets."""
    print("Testing validation performance...")
    
    validator = PathologyFLValidator()
    
    # Generate large dataset
    large_parameters = {}
    for i in range(100):
        large_parameters[f"layer_{i}"] = [0.001] * 1000
    
    # Test validation performance
    start_time = time.time()
    
    for _ in range(10):
        result = validator.validate_model_parameters(large_parameters)
    
    validation_time = time.time() - start_time
    
    # Test checksum performance
    start_time = time.time()
    
    for _ in range(100):
        checksum = validator.compute_checksum(large_parameters)
    
    checksum_time = time.time() - start_time
    
    print(f"  Large parameter validation: {validation_time:.4f}s")
    print(f"  Checksum computation: {checksum_time:.4f}s")
    print(f"  Parameters validated: {sum(len(p) for p in large_parameters.values()):,}")
    
    return validation_time < 1.0 and checksum_time < 1.0

def test_concurrent_validation():
    """Test validation under concurrent access."""
    print("Testing concurrent validation...")
    
    validator = PathologyFLValidator()
    
    from concurrent.futures import ThreadPoolExecutor
    
    def validate_worker(worker_id):
        """Worker function for concurrent validation."""
        metadata = {
            "hospital_type": "community",
            "years_experience": worker_id % 50,
            "diagnostic_accuracy": 0.8 + (worker_id % 20) / 100
        }
        
        result = validator.validate_hospital_metadata(metadata)
        return len(result["errors"]) == 0
    
    # Run concurrent validations
    num_workers = 20
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(validate_worker, range(num_workers)))
    
    concurrent_time = time.time() - start_time
    
    successful_validations = sum(results)
    
    print(f"  Concurrent workers: {num_workers}")
    print(f"  Successful validations: {successful_validations}")
    print(f"  Concurrent time: {concurrent_time:.4f}s")
    
    return successful_validations == num_workers and concurrent_time < 1.0

def run_pathology_fl_validation_tests():
    """Run all PathologyFL validation tests."""
    print("✅ PathologyFL Validation Testing")
    print("=" * 50)
    
    tests = [
        ("Metadata Validation", test_metadata_validation),
        ("Parameter Validation", test_parameter_validation),
        ("Aggregation Validation", test_aggregation_validation),
        ("Data Integrity", test_data_integrity),
        ("Performance Validation", test_performance_validation),
        ("Concurrent Validation", test_concurrent_validation),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        print()
    
    print("=" * 50)
    print(f"Validation Tests: {passed}/{len(tests)} passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_pathology_fl_validation_tests()
    exit(0 if success else 1)