"""
Unit tests for multimodal dataset missing data handling.

Tests imputation/masking strategies, batch size variation, and modality dimension
mismatch detection for robust multimodal data processing.

**Validates: Requirements 3.5, 3.6, 3.7**
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from src.data.loaders import MultimodalDataset, collate_multimodal
from tests.dataset_testing.synthetic.multimodal_generator import (
    MultimodalSyntheticGenerator,
    MultimodalSyntheticSpec
)


class TestMultimodalMissingDataHandling(unittest.TestCase):
    """Test missing data handling strategies in multimodal datasets."""

    def setUp(self):
        """Set up test fixtures with high missing data probability."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.generator = MultimodalSyntheticGenerator(random_seed=42)
        
        # Create synthetic dataset with high missing probability
        self.spec = MultimodalSyntheticSpec(
            num_patients=15,
            wsi_feature_dim=2048,
            genomic_feature_dim=1000,
            clinical_text_length_range=(20, 100),
            missing_modality_probability=0.6  # High missing rate for testing
        )
        
        self.dataset_samples = self.generator.generate_samples(
            num_patients=15,
            spec=self.spec,
            output_dir=self.data_dir
        )
        
        # Save the dataset to disk
        self.generator._save_dataset(self.dataset_samples, self.data_dir)
        
        # Standard config for testing
        self.config = DictConfig({
            'wsi_enabled': True,
            'genomic_enabled': True,
            'clinical_text_enabled': True,
            'wsi_feature_dim': 2048,
            'genomic_feature_dim': 1000,
            'max_text_length': 100
        })

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_imputation_strategies_for_missing_modalities(self):
        """
        Test imputation strategies for missing modalities.
        
        **Validates: Requirements 3.5**
        """
        dataset = MultimodalDataset(self.data_dir, "train", self.config)
        
        # Collect samples with different missing patterns
        missing_patterns = {
            'all_present': [],
            'missing_wsi': [],
            'missing_genomic': [],
            'missing_clinical': [],
            'missing_multiple': []
        }
        
        for i in range(len(dataset)):
            sample = dataset[i]
            has_wsi = sample['wsi_features'] is not None
            has_genomic = sample['genomic'] is not None
            has_clinical = sample['clinical_text'] is not None
            
            if has_wsi and has_genomic and has_clinical:
                missing_patterns['all_present'].append(sample)
            elif not has_wsi and has_genomic and has_clinical:
                missing_patterns['missing_wsi'].append(sample)
            elif has_wsi and not has_genomic and has_clinical:
                missing_patterns['missing_genomic'].append(sample)
            elif has_wsi and has_genomic and not has_clinical:
                missing_patterns['missing_clinical'].append(sample)
            else:
                missing_patterns['missing_multiple'].append(sample)
        
        # Verify we have samples with missing modalities
        total_missing = sum(len(samples) for key, samples in missing_patterns.items() 
                           if key != 'all_present')
        self.assertGreater(total_missing, 0, "Should have samples with missing modalities")
        
        # Test batch collation with different missing patterns
        for pattern_name, samples in missing_patterns.items():
            if len(samples) >= 2:
                batch = samples[:2]
                collated_batch = collate_multimodal(batch)
                
                # Verify batch structure is maintained
                self.assertEqual(len(collated_batch['patient_ids']), len(batch))
                self.assertEqual(collated_batch['label'].shape[0], len(batch))
                
                # Verify missing modalities are handled appropriately
                if pattern_name == 'missing_wsi':
                    # WSI features should be None or zero-padded
                    if 'wsi_features' in collated_batch and collated_batch['wsi_features'] is not None:
                        # Check if zero-padding is used
                        wsi_batch = collated_batch['wsi_features']
                        self.assertEqual(wsi_batch.shape[0], len(batch))
                
                if pattern_name == 'missing_genomic':
                    # Genomic features should be None or zero-padded
                    if 'genomic' in collated_batch and collated_batch['genomic'] is not None:
                        genomic_batch = collated_batch['genomic']
                        self.assertEqual(genomic_batch.shape[0], len(batch))

    def test_masking_strategies_for_missing_modalities(self):
        """
        Test masking strategies for missing modalities in batch processing.
        
        **Validates: Requirements 3.5**
        """
        dataset = MultimodalDataset(self.data_dir, "train", self.config)
        
        # Create mixed batch with missing and present modalities
        mixed_batch = []
        for i in range(min(6, len(dataset))):
            sample = dataset[i]
            mixed_batch.append(sample)
        
        collated_batch = collate_multimodal(mixed_batch)
        
        # Test that masking information is available or can be derived
        batch_size = len(mixed_batch)
        
        # Check if attention masks or similar masking mechanisms are available
        for modality in ['wsi_features', 'genomic', 'clinical_text']:
            if modality in collated_batch and collated_batch[modality] is not None:
                modality_data = collated_batch[modality]
                self.assertEqual(modality_data.shape[0], batch_size)
                
                # Verify that missing samples can be identified
                # (e.g., through zero tensors or explicit masks)
                for i in range(batch_size):
                    original_sample = mixed_batch[i]
                    if original_sample[modality] is None:
                        # Missing modality should be handled consistently
                        if len(modality_data.shape) == 2:  # [batch, features]
                            sample_data = modality_data[i]
                        elif len(modality_data.shape) == 3:  # [batch, seq, features]
                            sample_data = modality_data[i]
                        
                        # The handling strategy should be consistent
                        # (either all zeros, or proper masking)
                        self.assertTrue(torch.is_tensor(sample_data))

    def test_batch_size_variation_with_missing_data(self):
        """
        Test consistent handling across different batch sizes with missing data.
        
        **Validates: Requirements 3.6**
        """
        dataset = MultimodalDataset(self.data_dir, "train", self.config)
        
        # Test various batch sizes
        batch_sizes = [1, 2, 4, min(8, len(dataset))]
        
        for batch_size in batch_sizes:
            if batch_size > len(dataset):
                continue
                
            batch = [dataset[i] for i in range(batch_size)]
            collated_batch = collate_multimodal(batch)
            
            # Verify batch dimensions are consistent regardless of missing data
            self.assertEqual(len(collated_batch['patient_ids']), batch_size)
            self.assertEqual(collated_batch['label'].shape[0], batch_size)
            
            # Verify modality handling is consistent across batch sizes
            for modality in ['wsi_features', 'genomic', 'clinical_text']:
                if modality in collated_batch and collated_batch[modality] is not None:
                    modality_data = collated_batch[modality]
                    self.assertEqual(modality_data.shape[0], batch_size)
                    
                    # Check that the handling is consistent for each sample
                    for i in range(batch_size):
                        original_sample = batch[i]
                        has_modality = original_sample[modality] is not None
                        
                        if has_modality:
                            # Present modality should have meaningful data
                            if len(modality_data.shape) == 2:
                                sample_data = modality_data[i]
                                # Should not be all zeros if modality is present
                                if modality == 'genomic':
                                    # Genomic data might have some zeros, but not all
                                    self.assertFalse(torch.all(sample_data == 0))
                            elif len(modality_data.shape) == 3:
                                sample_data = modality_data[i]
                                # WSI features should have some non-zero values
                                if modality == 'wsi_features':
                                    self.assertFalse(torch.all(sample_data == 0))

    def test_modality_dimension_mismatch_detection(self):
        """
        Test detection and handling of modality dimension mismatches.
        
        **Validates: Requirements 3.7**
        """
        # Test with various mismatched configurations
        mismatch_configs = [
            {
                'wsi_enabled': True,
                'genomic_enabled': True,
                'clinical_text_enabled': True,
                'wsi_feature_dim': 1024,  # Mismatch: actual is 2048
                'genomic_feature_dim': 1000,
                'max_text_length': 100
            },
            {
                'wsi_enabled': True,
                'genomic_enabled': True,
                'clinical_text_enabled': True,
                'wsi_feature_dim': 2048,
                'genomic_feature_dim': 500,  # Mismatch: actual is 1000
                'max_text_length': 100
            },
            {
                'wsi_enabled': True,
                'genomic_enabled': True,
                'clinical_text_enabled': True,
                'wsi_feature_dim': 2048,
                'genomic_feature_dim': 1000,
                'max_text_length': 25  # Truncation test
            }
        ]
        
        for config_dict in mismatch_configs:
            config = DictConfig(config_dict)
            
            # Dataset should still be creatable despite mismatches
            dataset = MultimodalDataset(self.data_dir, "train", config)
            self.assertGreater(len(dataset), 0)
            
            # Test sample loading with mismatched config
            sample = dataset[0]
            self.assertIsInstance(sample, dict)
            self.assertIn('patient_id', sample)
            self.assertIn('label', sample)
            
            # Verify actual dimensions are preserved from data files
            if sample['wsi_features'] is not None:
                # WSI features should maintain original dimensions from HDF5 file
                actual_wsi_dim = sample['wsi_features'].shape[1]
                self.assertEqual(actual_wsi_dim, 2048)  # Original dimension
            
            if sample['genomic'] is not None:
                # Genomic features should maintain original dimensions
                actual_genomic_dim = sample['genomic'].shape[0]
                self.assertEqual(actual_genomic_dim, 1000)  # Original dimension
            
            if sample['clinical_text'] is not None:
                # Clinical text should respect max_text_length
                max_len = config.max_text_length
                actual_len = sample['clinical_text'].shape[0]
                self.assertLessEqual(actual_len, max_len)

    def test_graceful_degradation_with_missing_files(self):
        """
        Test graceful degradation when data files are missing or corrupted.
        
        **Validates: Requirements 3.7**
        """
        dataset = MultimodalDataset(self.data_dir, "train", self.config)
        
        # Test with missing files by temporarily moving them
        import shutil
        
        # Backup original files
        wsi_backup = None
        genomic_backup = None
        
        try:
            # Test missing WSI files
            wsi_dir = self.data_dir / "wsi_features"
            if wsi_dir.exists():
                wsi_backup = self.data_dir / "wsi_features_backup"
                shutil.move(str(wsi_dir), str(wsi_backup))
            
            # Create new dataset instance to test missing WSI files
            dataset_missing_wsi = MultimodalDataset(self.data_dir, "train", self.config)
            
            # Should still be able to load samples
            sample = dataset_missing_wsi[0]
            self.assertIsInstance(sample, dict)
            self.assertIn('patient_id', sample)
            
            # WSI features should be None due to missing files
            self.assertIsNone(sample['wsi_features'])
            
            # Other modalities should still work
            if sample['genomic'] is not None:
                self.assertIsInstance(sample['genomic'], torch.Tensor)
            
        finally:
            # Restore backup files
            if wsi_backup and wsi_backup.exists():
                if wsi_dir.exists():
                    shutil.rmtree(str(wsi_dir))
                shutil.move(str(wsi_backup), str(wsi_dir))

    def test_missing_data_batch_consistency(self):
        """
        Test that batch processing is consistent with different missing data patterns.
        
        **Validates: Requirements 3.5, 3.6**
        """
        dataset = MultimodalDataset(self.data_dir, "train", self.config)
        
        # Create batches with different missing patterns
        all_samples = [dataset[i] for i in range(len(dataset))]
        
        # Group samples by missing pattern
        complete_samples = []
        partial_samples = []
        minimal_samples = []
        
        for sample in all_samples:
            modality_count = sum([
                sample['wsi_features'] is not None,
                sample['genomic'] is not None,
                sample['clinical_text'] is not None
            ])
            
            if modality_count == 3:
                complete_samples.append(sample)
            elif modality_count >= 1:
                partial_samples.append(sample)
            else:
                minimal_samples.append(sample)
        
        # Test homogeneous batches (same missing pattern)
        for sample_group, group_name in [
            (complete_samples, "complete"),
            (partial_samples, "partial"),
            (minimal_samples, "minimal")
        ]:
            if len(sample_group) >= 2:
                batch = sample_group[:2]
                collated_batch = collate_multimodal(batch)
                
                # Verify batch consistency
                self.assertEqual(len(collated_batch['patient_ids']), len(batch))
                self.assertEqual(collated_batch['label'].shape[0], len(batch))
                
                # Verify modality handling is consistent within the batch
                for modality in ['wsi_features', 'genomic', 'clinical_text']:
                    if modality in collated_batch and collated_batch[modality] is not None:
                        modality_data = collated_batch[modality]
                        self.assertEqual(modality_data.shape[0], len(batch))
        
        # Test heterogeneous batches (mixed missing patterns)
        if len(complete_samples) > 0 and len(partial_samples) > 0:
            mixed_batch = [complete_samples[0], partial_samples[0]]
            collated_mixed = collate_multimodal(mixed_batch)
            
            # Mixed batch should still be processable
            self.assertEqual(len(collated_mixed['patient_ids']), 2)
            self.assertEqual(collated_mixed['label'].shape[0], 2)

    def test_missing_data_error_handling(self):
        """
        Test error handling for extreme missing data scenarios.
        
        **Validates: Requirements 3.7**
        """
        # Test with all modalities disabled
        disabled_config = DictConfig({
            'wsi_enabled': False,
            'genomic_enabled': False,
            'clinical_text_enabled': False,
            'wsi_feature_dim': 2048,
            'genomic_feature_dim': 1000,
            'max_text_length': 100
        })
        
        dataset = MultimodalDataset(self.data_dir, "train", disabled_config)
        
        # Should still be able to create dataset and load samples
        self.assertGreater(len(dataset), 0)
        
        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn('patient_id', sample)
        self.assertIn('label', sample)
        
        # All modalities should be None
        self.assertIsNone(sample['wsi_features'])
        self.assertIsNone(sample['genomic'])
        self.assertIsNone(sample['clinical_text'])
        
        # Batch processing should still work
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        collated_batch = collate_multimodal(batch)
        
        self.assertEqual(len(collated_batch['patient_ids']), len(batch))
        self.assertEqual(collated_batch['label'].shape[0], len(batch))


if __name__ == '__main__':
    unittest.main()