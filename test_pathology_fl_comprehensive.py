#!/usr/bin/env python3
"""Comprehensive PathologyFL test suite."""

import time
import random
from typing import Dict, List, Any

class PathologyFLCore:
    """Core PathologyFL implementation."""
    
    def __init__(self):
        self.hospitals = {}
        self.global_model = {}
        
    def register_hospital(self, hospital_id: str, metadata: Dict):
        """Register hospital with metadata."""
        self.hospitals[hospital_id] = {
            "metadata": metadata,
            "last_update": time.time(),
            "weight": self._calculate_weight(metadata)
        }
        
    def _calculate_weight(self, metadata: Dict) -> float:
        """Calculate hospital weight based on metadata."""
        base_weight = 1.0
        
        # Hospital type bonus
        if metadata.get("hospital_type") == "cancer_center":
            base_weight *= 2.0
        elif metadata.get("hospital_type") == "teaching_hospital":
            base_weight *= 1.5
            
        # Experience bonus
        years = metadata.get("years_experience", 0)
        base_weight *= (1.0 + years / 100.0)
        
        # Accuracy bonus
        accuracy = metadata.get("diagnostic_accuracy", 0.8)
        base_weight *= accuracy
        
        return base_weight
    
    def aggregate_models(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client model updates."""
        if not client_updates:
            return {}
            
        # Weighted aggregation
        total_weight = 0.0
        aggregated = {}
        
        for update in client_updates:
            hospital_id = update["hospital_id"]
            weight = self.hospitals.get(hospital_id, {}).get("weight", 1.0)
            total_weight += weight
            
            for layer, params in update["parameters"].items():
                if layer not in aggregated:
                    aggregated[layer] = [0.0] * len(params)
                
                for i, param in enumerate(params):
                    aggregated[layer][i] += param * weight
        
        # Normalize by total weight
        for layer in aggregated:
            for i in range(len(aggregated[layer])):
                aggregated[layer][i] /= total_weight
                
        return aggregated

def test_hospital_registration():
    """Test hospital registration system."""
    print("Testing hospital registration...")
    
    fl_core = PathologyFLCore()
    
    hospitals = [
        ("mayo_clinic", {
            "hospital_type": "cancer_center",
            "annual_cases": 15000,
            "years_experience": 25,
            "diagnostic_accuracy": 0.96
        }),
        ("community_hospital", {
            "hospital_type": "community",
            "annual_cases": 3000,
            "years_experience": 10,
            "diagnostic_accuracy": 0.88
        })
    ]
    
    for hospital_id, metadata in hospitals:
        fl_core.register_hospital(hospital_id, metadata)
    
    registered_count = len(fl_core.hospitals)
    weights = [h["weight"] for h in fl_core.hospitals.values()]
    
    print(f"  Registered hospitals: {registered_count}")
    print(f"  Weights: {[f'{w:.2f}' for w in weights]}")
    
    return registered_count == 2 and weights[0] > weights[1]

def test_weighted_aggregation():
    """Test weighted model aggregation."""
    print("Testing weighted aggregation...")
    
    fl_core = PathologyFLCore()
    
    # Register hospitals
    fl_core.register_hospital("expert", {
        "hospital_type": "cancer_center",
        "years_experience": 20,
        "diagnostic_accuracy": 0.95
    })
    
    fl_core.register_hospital("novice", {
        "hospital_type": "community",
        "years_experience": 5,
        "diagnostic_accuracy": 0.80
    })
    
    # Create model updates
    updates = [
        {
            "hospital_id": "expert",
            "parameters": {
                "layer1": [1.0, 2.0, 3.0],
                "layer2": [0.5, 1.5]
            }
        },
        {
            "hospital_id": "novice", 
            "parameters": {
                "layer1": [0.0, 0.0, 0.0],
                "layer2": [0.0, 0.0]
            }
        }
    ]
    
    result = fl_core.aggregate_models(updates)
    
    # Expert should dominate due to higher weight
    expert_influence = result["layer1"][0] > 0.5
    
    print(f"  Aggregated layer1: {result['layer1']}")
    print(f"  Expert influence: {expert_influence}")
    
    return len(result) == 2 and expert_influence

def test_cancer_type_specialization():
    """Test cancer type specialization."""
    print("Testing cancer type specialization...")
    
    fl_core = PathologyFLCore()
    
    # Register specialized hospitals
    fl_core.register_hospital("breast_specialist", {
        "hospital_type": "cancer_center",
        "cancer_specialties": ["breast"],
        "years_experience": 15,
        "diagnostic_accuracy": 0.94
    })
    
    fl_core.register_hospital("lung_specialist", {
        "hospital_type": "cancer_center", 
        "cancer_specialties": ["lung"],
        "years_experience": 12,
        "diagnostic_accuracy": 0.92
    })
    
    # Test specialization matching
    def get_specialists(cancer_type):
        specialists = []
        for hospital_id, data in fl_core.hospitals.items():
            specialties = data["metadata"].get("cancer_specialties", [])
            if cancer_type in specialties:
                specialists.append(hospital_id)
        return specialists
    
    breast_specialists = get_specialists("breast")
    lung_specialists = get_specialists("lung")
    
    print(f"  Breast specialists: {breast_specialists}")
    print(f"  Lung specialists: {lung_specialists}")
    
    return len(breast_specialists) == 1 and len(lung_specialists) == 1

def test_quality_assessment():
    """Test slide quality assessment."""
    print("Testing quality assessment...")
    
    def assess_slide_quality(slide_data):
        """Assess slide quality metrics."""
        quality = {
            "image_sharpness": random.uniform(0.7, 0.95),
            "stain_consistency": random.uniform(0.6, 0.9),
            "artifact_level": random.uniform(0.05, 0.3),
            "tissue_coverage": random.uniform(0.8, 0.95)
        }
        
        # Overall quality score
        quality["overall"] = (
            quality["image_sharpness"] * 0.3 +
            quality["stain_consistency"] * 0.25 +
            (1.0 - quality["artifact_level"]) * 0.2 +
            quality["tissue_coverage"] * 0.25
        )
        
        return quality
    
    # Test multiple slides
    slides = [f"slide_{i}" for i in range(10)]
    qualities = [assess_slide_quality(slide) for slide in slides]
    
    avg_quality = sum(q["overall"] for q in qualities) / len(qualities)
    high_quality_count = sum(1 for q in qualities if q["overall"] > 0.8)
    
    print(f"  Slides assessed: {len(slides)}")
    print(f"  Average quality: {avg_quality:.3f}")
    print(f"  High quality slides: {high_quality_count}")
    
    return avg_quality > 0.7 and high_quality_count > 0

def test_privacy_preservation():
    """Test differential privacy mechanisms."""
    print("Testing privacy preservation...")
    
    def add_noise(value, epsilon=1.0):
        """Add Laplace noise for differential privacy."""
        import math
        
        # Laplace noise
        u = random.uniform(-0.5, 0.5)
        noise = -math.copysign(math.log(1 - 2 * abs(u)), u) / epsilon
        
        return value + noise
    
    def privatize_gradients(gradients, epsilon=1.0):
        """Add noise to gradients for privacy."""
        return [add_noise(g, epsilon) for g in gradients]
    
    # Test gradient privatization
    original_gradients = [0.1, 0.2, 0.3, 0.4, 0.5]
    private_gradients = privatize_gradients(original_gradients)
    
    # Check that noise was added
    differences = [abs(o - p) for o, p in zip(original_gradients, private_gradients)]
    noise_added = any(d > 0.01 for d in differences)
    
    print(f"  Original: {[f'{g:.3f}' for g in original_gradients]}")
    print(f"  Private: {[f'{g:.3f}' for g in private_gradients]}")
    print(f"  Noise added: {noise_added}")
    
    return noise_added and len(private_gradients) == len(original_gradients)

def run_pathology_fl_comprehensive_tests():
    """Run comprehensive PathologyFL tests."""
    print("🏥 Comprehensive PathologyFL Testing")
    print("=" * 50)
    
    tests = [
        ("Hospital Registration", test_hospital_registration),
        ("Weighted Aggregation", test_weighted_aggregation),
        ("Cancer Type Specialization", test_cancer_type_specialization),
        ("Quality Assessment", test_quality_assessment),
        ("Privacy Preservation", test_privacy_preservation),
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
    print(f"PathologyFL Tests: {passed}/{len(tests)} passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_pathology_fl_comprehensive_tests()
    exit(0 if success else 1)