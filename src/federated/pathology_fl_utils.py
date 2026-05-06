#!/usr/bin/env python3
"""
PathologyFL Utilities - Helper functions for medical expertise FL
"""

from typing import Dict, List
from src.federated.pathology_fl import HospitalType, CancerType

def create_hospital_network(hospital_configs: List[Dict]) -> Dict:
    """Create a network of hospitals from configuration."""
    
    network = {}
    for config in hospital_configs:
        hospital_id = config['hospital_id']
        network[hospital_id] = {
            'hospital_type': config.get('hospital_type', 'community_hospital'),
            'annual_cases': config.get('annual_cases', 5000),
            'cancer_specialties': config.get('cancer_specialties', ['general']),
            'diagnostic_accuracy': config.get('diagnostic_accuracy', 0.85),
            'years_experience': config.get('years_experience', 10)
        }
    
    return network

def validate_hospital_metadata(metadata: Dict) -> bool:
    """Validate hospital metadata for PathologyFL."""
    
    required_fields = [
        'hospital_type', 'annual_cases', 'cancer_specialties',
        'diagnostic_accuracy', 'years_experience'
    ]
    
    for field in required_fields:
        if field not in metadata:
            return False
    
    # Validate ranges
    if not (0.0 <= metadata['diagnostic_accuracy'] <= 1.0):
        return False
    
    if metadata['annual_cases'] < 0:
        return False
    
    if metadata['years_experience'] < 0:
        return False
    
    return True

def estimate_fl_performance(num_hospitals: int, avg_cases: int) -> Dict:
    """Estimate PathologyFL performance metrics."""
    
    return {
        'estimated_rounds': max(10, num_hospitals // 2),
        'convergence_time': f"{num_hospitals * 0.5:.1f} hours",
        'data_efficiency': min(1.0, avg_cases / 10000),
        'expertise_diversity': min(1.0, num_hospitals / 20)
    }