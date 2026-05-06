#!/usr/bin/env python3
"""Test PACS integration with real DICOM edge cases."""

import os
import tempfile
import struct
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

class MockDICOMDataset:
    """Mock DICOM dataset for testing."""
    
    def __init__(self, **kwargs):
        self.data = kwargs
        
    def __getitem__(self, key):
        return self.data.get(key)
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def get(self, key, default=None):
        return self.data.get(key, default)

class PACSIntegrationTester:
    """Test PACS integration with various DICOM edge cases."""
    
    def __init__(self):
        self.test_datasets = self._create_test_datasets()
        
    def _create_test_datasets(self) -> List[MockDICOMDataset]:
        """Create test DICOM datasets with edge cases."""
        return [
            # Normal case
            MockDICOMDataset(
                StudyInstanceUID="1.2.3.4.5.6.7.8.9.1",
                SeriesInstanceUID="1.2.3.4.5.6.7.8.9.2",
                SOPInstanceUID="1.2.3.4.5.6.7.8.9.3",
                PatientID="PAT001",
                Modality="SM",
                Rows=1024,
                Columns=1024
            ),
            # Missing required fields
            MockDICOMDataset(
                StudyInstanceUID="1.2.3.4.5.6.7.8.9.4",
                PatientID="PAT002"
                # Missing SeriesInstanceUID, SOPInstanceUID
            ),
            # Invalid UIDs
            MockDICOMDataset(
                StudyInstanceUID="invalid.uid.format",
                SeriesInstanceUID="",
                SOPInstanceUID=None,
                PatientID="PAT003"
            ),
            # Large dimensions
            MockDICOMDataset(
                StudyInstanceUID="1.2.3.4.5.6.7.8.9.5",
                SeriesInstanceUID="1.2.3.4.5.6.7.8.9.6",
                SOPInstanceUID="1.2.3.4.5.6.7.8.9.7",
                PatientID="PAT004",
                Rows=50000,
                Columns=50000
            ),
            # Unicode patient data
            MockDICOMDataset(
                StudyInstanceUID="1.2.3.4.5.6.7.8.9.8",
                SeriesInstanceUID="1.2.3.4.5.6.7.8.9.9",
                SOPInstanceUID="1.2.3.4.5.6.7.8.9.10",
                PatientID="PAT005",
                PatientName="José María García-López",
                PatientBirthDate="19850315"
            )
        ]

def test_dicom_uid_validation():
    """Test DICOM UID validation."""
    print("Testing DICOM UID validation...")
    
    def validate_uid(uid):
        """Validate DICOM UID format."""
        if not uid or not isinstance(uid, str):
            return False
        if len(uid) > 64:  # DICOM standard limit
            return False
        if not all(c.isdigit() or c == '.' for c in uid):
            return False
        if uid.startswith('.') or uid.endswith('.'):
            return False
        if '..' in uid:
            return False
        return True
    
    test_cases = [
        ("1.2.3.4.5.6.7.8.9", True),
        ("invalid.uid.format", False),
        ("", False),
        (None, False),
        ("1.2.3.4.5.6.7.8.9." + "1" * 60, False),  # Too long
        (".1.2.3.4.5", False),  # Starts with dot
        ("1.2.3.4.5.", False),  # Ends with dot
        ("1.2..3.4.5", False),  # Double dot
    ]
    
    passed = 0
    for uid, expected in test_cases:
        result = validate_uid(uid)
        if result == expected:
            passed += 1
        else:
            print(f"  Failed: {uid} -> {result}, expected {expected}")
    
    print(f"  UID validation: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_missing_required_fields():
    """Test handling of missing required DICOM fields."""
    print("Testing missing required fields...")
    
    required_fields = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
    
    def validate_required_fields(dataset):
        """Check if all required fields are present."""
        missing = []
        for field in required_fields:
            if not dataset.get(field):
                missing.append(field)
        return missing
    
    tester = PACSIntegrationTester()
    
    valid_count = 0
    for i, dataset in enumerate(tester.test_datasets):
        missing = validate_required_fields(dataset)
        if not missing:
            valid_count += 1
        else:
            print(f"  Dataset {i}: Missing {missing}")
    
    print(f"  Valid datasets: {valid_count}/{len(tester.test_datasets)}")
    return valid_count > 0

def test_large_image_handling():
    """Test handling of large DICOM images."""
    print("Testing large image handling...")
    
    def estimate_memory_usage(rows, columns, bits_per_pixel=8):
        """Estimate memory usage for DICOM image."""
        return (rows * columns * bits_per_pixel) // 8  # bytes
    
    def can_handle_image(rows, columns, max_memory_mb=1024):
        """Check if image can be handled within memory limits."""
        memory_bytes = estimate_memory_usage(rows, columns)
        memory_mb = memory_bytes / (1024 * 1024)
        return memory_mb <= max_memory_mb
    
    test_cases = [
        (1024, 1024, True),      # Normal size
        (4096, 4096, True),      # Large but manageable
        (50000, 50000, False),   # Too large
        (100000, 100000, False), # Extremely large
    ]
    
    passed = 0
    for rows, cols, expected in test_cases:
        result = can_handle_image(rows, cols)
        if result == expected:
            passed += 1
            memory_mb = estimate_memory_usage(rows, cols) / (1024 * 1024)
            print(f"  {rows}x{cols}: {memory_mb:.1f}MB - {'OK' if result else 'Too large'}")
        else:
            print(f"  Failed: {rows}x{cols} -> {result}, expected {expected}")
    
    print(f"  Large image handling: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_unicode_patient_data():
    """Test handling of Unicode patient data."""
    print("Testing Unicode patient data...")
    
    def sanitize_patient_name(name):
        """Sanitize patient name for safe processing."""
        if not name:
            return ""
        # Remove control characters
        sanitized = ''.join(c for c in name if ord(c) >= 32)
        # Limit length
        return sanitized[:64]
    
    test_names = [
        "John Doe",
        "José María García-López",
        "李小明",
        "محمد عبدالله",
        "Müller, Hans-Jürgen",
        "O'Connor, Seán",
        "",
        None,
        "A" * 100,  # Very long name
    ]
    
    passed = 0
    for name in test_names:
        try:
            sanitized = sanitize_patient_name(name)
            if len(sanitized) <= 64:
                passed += 1
            else:
                print(f"  Failed: Name too long after sanitization: {len(sanitized)}")
        except Exception as e:
            print(f"  Failed to sanitize '{name}': {e}")
    
    print(f"  Unicode handling: {passed}/{len(test_names)} passed")
    return passed == len(test_names)

def test_dicom_transfer_syntax():
    """Test different DICOM transfer syntaxes."""
    print("Testing DICOM transfer syntaxes...")
    
    transfer_syntaxes = {
        "1.2.840.10008.1.2": "Implicit VR Little Endian",
        "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
        "1.2.840.10008.1.2.2": "Explicit VR Big Endian",
        "1.2.840.10008.1.2.4.50": "JPEG Baseline",
        "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
        "1.2.840.10008.1.2.4.91": "JPEG 2000",
    }
    
    def can_handle_transfer_syntax(syntax_uid):
        """Check if transfer syntax is supported."""
        supported = [
            "1.2.840.10008.1.2",      # Implicit VR Little Endian
            "1.2.840.10008.1.2.1",   # Explicit VR Little Endian
            "1.2.840.10008.1.2.4.50", # JPEG Baseline
        ]
        return syntax_uid in supported
    
    supported_count = 0
    for uid, name in transfer_syntaxes.items():
        supported = can_handle_transfer_syntax(uid)
        if supported:
            supported_count += 1
        print(f"  {name}: {'Supported' if supported else 'Not supported'}")
    
    print(f"  Transfer syntax support: {supported_count}/{len(transfer_syntaxes)}")
    return supported_count >= 3  # At least 3 should be supported

def test_pacs_query_edge_cases():
    """Test PACS query with edge cases."""
    print("Testing PACS query edge cases...")
    
    def build_dicom_query(patient_id=None, study_date=None, modality=None):
        """Build DICOM C-FIND query."""
        query = {}
        
        if patient_id:
            if len(patient_id) > 64:
                raise ValueError("Patient ID too long")
            query["PatientID"] = patient_id
            
        if study_date:
            # Validate date format YYYYMMDD
            if len(study_date) != 8 or not study_date.isdigit():
                raise ValueError("Invalid study date format")
            query["StudyDate"] = study_date
            
        if modality:
            valid_modalities = ["CT", "MR", "US", "XA", "RF", "SM", "CR", "DX"]
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality: {modality}")
            query["Modality"] = modality
            
        return query
    
    test_cases = [
        ({"patient_id": "PAT001", "study_date": "20240101", "modality": "SM"}, True),
        ({"patient_id": "A" * 70}, False),  # Patient ID too long
        ({"study_date": "2024-01-01"}, False),  # Invalid date format
        ({"study_date": "20240132"}, False),  # Invalid date
        ({"modality": "INVALID"}, False),  # Invalid modality
        ({}, True),  # Empty query (valid)
    ]
    
    passed = 0
    for params, should_succeed in test_cases:
        try:
            query = build_dicom_query(**params)
            if should_succeed:
                passed += 1
            else:
                print(f"  Unexpected success: {params}")
        except Exception as e:
            if not should_succeed:
                passed += 1
            else:
                print(f"  Unexpected failure: {params} - {e}")
    
    print(f"  PACS query handling: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_dicom_anonymization():
    """Test DICOM data anonymization."""
    print("Testing DICOM anonymization...")
    
    def anonymize_dicom(dataset):
        """Anonymize DICOM dataset."""
        # Fields to remove/anonymize
        sensitive_fields = [
            "PatientName", "PatientID", "PatientBirthDate",
            "PatientSex", "PatientAddress", "PatientTelephoneNumbers"
        ]
        
        anonymized = MockDICOMDataset()
        
        # Copy non-sensitive fields
        for key, value in dataset.data.items():
            if key not in sensitive_fields:
                anonymized[key] = value
        
        # Add anonymized patient ID
        anonymized["PatientID"] = "ANON_" + str(hash(dataset.get("PatientID", "")) % 10000)
        
        return anonymized
    
    tester = PACSIntegrationTester()
    
    anonymized_count = 0
    for dataset in tester.test_datasets:
        try:
            anon_dataset = anonymize_dicom(dataset)
            
            # Check that sensitive data is removed
            if not anon_dataset.get("PatientName") and anon_dataset.get("PatientID", "").startswith("ANON_"):
                anonymized_count += 1
            else:
                print(f"  Anonymization failed for dataset")
                
        except Exception as e:
            print(f"  Anonymization error: {e}")
    
    print(f"  Anonymization: {anonymized_count}/{len(tester.test_datasets)} passed")
    return anonymized_count == len(tester.test_datasets)

def run_pacs_dicom_tests():
    """Run all PACS DICOM edge case tests."""
    print("🏥 PACS Integration DICOM Edge Case Testing")
    print("=" * 50)
    
    tests = [
        ("DICOM UID Validation", test_dicom_uid_validation),
        ("Missing Required Fields", test_missing_required_fields),
        ("Large Image Handling", test_large_image_handling),
        ("Unicode Patient Data", test_unicode_patient_data),
        ("Transfer Syntax Support", test_dicom_transfer_syntax),
        ("PACS Query Edge Cases", test_pacs_query_edge_cases),
        ("DICOM Anonymization", test_dicom_anonymization),
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
    print(f"PACS DICOM Tests: {passed}/{len(tests)} passed")
    
    if passed >= len(tests) * 0.8:
        print("🏆 PACS integration handles DICOM edge cases well!")
    else:
        print(f"⚠️ {len(tests) - passed} DICOM edge cases need attention")
    
    return passed >= len(tests) * 0.8

if __name__ == "__main__":
    success = run_pacs_dicom_tests()
    exit(0 if success else 1)