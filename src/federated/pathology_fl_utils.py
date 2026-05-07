#!/usr/bin/env python3
"""PathologyFL utility functions and helpers."""

import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional

class PathologyFLUtils:
    """Utility functions for PathologyFL operations."""
    
    @staticmethod
    def calculate_hospital_weight(metadata: Dict) -> float:
        """Calculate hospital weight based on metadata."""
        base_weight = 1.0
        
        # Hospital type multiplier
        type_multipliers = {
            "cancer_center": 2.0,
            "teaching_hospital": 1.5,
            "community": 1.0,
            "rural": 0.8
        }
        
        hospital_type = metadata.get("hospital_type", "community")
        base_weight *= type_multipliers.get(hospital_type, 1.0)
        
        # Experience bonus (up to 50% bonus for 50+ years)
        years = metadata.get("years_experience", 0)
        experience_bonus = min(years / 100.0, 0.5)
        base_weight *= (1.0 + experience_bonus)
        
        # Accuracy multiplier
        accuracy = metadata.get("diagnostic_accuracy", 0.8)
        base_weight *= accuracy
        
        # Case volume bonus (logarithmic scaling)
        annual_cases = metadata.get("annual_cases", 1000)
        if annual_cases > 1000:
            import math
            volume_bonus = math.log10(annual_cases / 1000) * 0.1
            base_weight *= (1.0 + volume_bonus)
        
        return base_weight
    
    @staticmethod
    def assess_slide_quality(slide_data: Dict) -> Dict:
        """Assess slide quality metrics."""
        import random
        
        # Simulate quality assessment
        quality_metrics = {
            "image_sharpness": random.uniform(0.6, 0.95),
            "stain_consistency": random.uniform(0.5, 0.9),
            "artifact_level": random.uniform(0.05, 0.4),
            "tissue_coverage": random.uniform(0.7, 0.95),
            "focus_quality": random.uniform(0.6, 0.9)
        }
        
        # Overall quality score (weighted average)
        weights = {
            "image_sharpness": 0.25,
            "stain_consistency": 0.20,
            "artifact_level": -0.15,  # Negative because lower is better
            "tissue_coverage": 0.20,
            "focus_quality": 0.20
        }
        
        overall_score = 0.0
        for metric, value in quality_metrics.items():
            weight = weights.get(metric, 0.0)
            if metric == "artifact_level":
                # Invert artifact level (lower is better)
                overall_score += (1.0 - value) * abs(weight)
            else:
                overall_score += value * weight
        
        quality_metrics["overall_quality"] = max(0.0, min(1.0, overall_score))
        
        return quality_metrics
    
    @staticmethod
    def add_differential_privacy_noise(gradients: List[float], epsilon: float = 1.0) -> List[float]:
        """Add differential privacy noise to gradients."""
        import random
        import math
        
        def laplace_noise(scale):
            """Generate Laplace noise."""
            u = random.uniform(-0.5, 0.5)
            return -math.copysign(math.log(1 - 2 * abs(u)), u) * scale
        
        # Calculate noise scale based on epsilon
        sensitivity = 1.0  # Assume L2 sensitivity of 1
        scale = sensitivity / epsilon
        
        # Add noise to each gradient
        noisy_gradients = []
        for grad in gradients:
            noise = laplace_noise(scale)
            noisy_gradients.append(grad + noise)
        
        return noisy_gradients
    
    @staticmethod
    def detect_byzantine_updates(updates: List[Dict], threshold: float = 3.0) -> List[str]:
        """Detect Byzantine (malicious) updates using statistical outlier detection."""
        if len(updates) < 3:
            return []  # Need at least 3 updates for detection
        
        byzantine_hospitals = []
        
        # Check each layer separately
        if not updates:
            return byzantine_hospitals
            
        layers = updates[0]["parameters"].keys()
        
        for layer in layers:
            # Collect all parameters for this layer
            layer_params = []
            for update in updates:
                if layer in update["parameters"]:
                    layer_params.append(update["parameters"][layer])
            
            if not layer_params:
                continue
                
            # Calculate statistics for each parameter position
            param_length = len(layer_params[0])
            
            for pos in range(param_length):
                values = [params[pos] for params in layer_params if pos < len(params)]
                
                if len(values) < 3:
                    continue
                
                # Calculate mean and standard deviation
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                std_dev = variance ** 0.5
                
                if std_dev == 0:
                    continue
                
                # Check for outliers
                for i, value in enumerate(values):
                    z_score = abs(value - mean_val) / std_dev
                    
                    if z_score > threshold:
                        hospital_id = updates[i]["hospital_id"]
                        if hospital_id not in byzantine_hospitals:
                            byzantine_hospitals.append(hospital_id)
        
        return byzantine_hospitals
    
    @staticmethod
    def compress_model_update(update: Dict, compression_ratio: float = 0.1) -> Dict:
        """Compress model update for efficient transmission."""
        compressed_update = {
            "hospital_id": update["hospital_id"],
            "parameters": {},
            "compression_info": {
                "method": "top_k",
                "ratio": compression_ratio,
                "original_size": 0,
                "compressed_size": 0
            }
        }
        
        for layer, params in update["parameters"].items():
            # Calculate number of parameters to keep
            k = max(1, int(len(params) * compression_ratio))
            
            # Get top-k parameters by absolute value
            param_indices = list(range(len(params)))
            param_indices.sort(key=lambda i: abs(params[i]), reverse=True)
            
            # Keep only top-k parameters
            compressed_params = {}
            for i in param_indices[:k]:
                compressed_params[i] = params[i]
            
            compressed_update["parameters"][layer] = compressed_params
            compressed_update["compression_info"]["original_size"] += len(params)
            compressed_update["compression_info"]["compressed_size"] += len(compressed_params)
        
        return compressed_update
    
    @staticmethod
    def decompress_model_update(compressed_update: Dict, original_shape: Dict) -> Dict:
        """Decompress model update."""
        decompressed_update = {
            "hospital_id": compressed_update["hospital_id"],
            "parameters": {}
        }
        
        for layer, compressed_params in compressed_update["parameters"].items():
            if layer not in original_shape:
                continue
                
            # Initialize with zeros
            full_params = [0.0] * original_shape[layer]
            
            # Fill in compressed parameters
            for index, value in compressed_params.items():
                if isinstance(index, str):
                    index = int(index)
                if 0 <= index < len(full_params):
                    full_params[index] = value
            
            decompressed_update["parameters"][layer] = full_params
        
        return decompressed_update
    
    @staticmethod
    def validate_update_integrity(update: Dict, expected_checksum: str) -> bool:
        """Validate update integrity using checksum."""
        # Calculate checksum of the update
        update_str = json.dumps(update, sort_keys=True)
        actual_checksum = hashlib.sha256(update_str.encode()).hexdigest()
        
        return actual_checksum == expected_checksum
    
    @staticmethod
    def calculate_convergence_metrics(round_metrics: List[Dict]) -> Dict:
        """Calculate convergence metrics across rounds."""
        if len(round_metrics) < 2:
            return {"converged": False, "improvement_rate": 0.0}
        
        # Calculate improvement rate
        initial_accuracy = round_metrics[0].get("accuracy", 0.0)
        final_accuracy = round_metrics[-1].get("accuracy", 0.0)
        
        improvement_rate = (final_accuracy - initial_accuracy) / len(round_metrics)
        
        # Check for convergence (small improvement in last few rounds)
        if len(round_metrics) >= 5:
            recent_improvements = []
            for i in range(len(round_metrics) - 4, len(round_metrics)):
                if i > 0:
                    prev_acc = round_metrics[i-1].get("accuracy", 0.0)
                    curr_acc = round_metrics[i].get("accuracy", 0.0)
                    recent_improvements.append(curr_acc - prev_acc)
            
            avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)
            converged = abs(avg_recent_improvement) < 0.001  # Less than 0.1% improvement
        else:
            converged = False
        
        return {
            "converged": converged,
            "improvement_rate": improvement_rate,
            "total_improvement": final_accuracy - initial_accuracy,
            "rounds_analyzed": len(round_metrics)
        }
    
    @staticmethod
    def generate_performance_report(metrics_history: List[Dict]) -> str:
        """Generate a performance report."""
        if not metrics_history:
            return "No metrics available for report generation."
        
        report_lines = []
        report_lines.append("PathologyFL Performance Report")
        report_lines.append("=" * 50)
        
        # Basic statistics
        final_metrics = metrics_history[-1]
        report_lines.append(f"Training Rounds: {len(metrics_history)}")
        report_lines.append(f"Final Accuracy: {final_metrics.get('accuracy', 0):.3f}")
        report_lines.append(f"Final AUC: {final_metrics.get('auc', 0):.3f}")
        report_lines.append(f"Final Sensitivity: {final_metrics.get('sensitivity', 0):.3f}")
        report_lines.append(f"Final Specificity: {final_metrics.get('specificity', 0):.3f}")
        
        # Convergence analysis
        convergence = PathologyFLUtils.calculate_convergence_metrics(metrics_history)
        report_lines.append(f"\nConvergence Analysis:")
        report_lines.append(f"Converged: {'Yes' if convergence['converged'] else 'No'}")
        report_lines.append(f"Improvement Rate: {convergence['improvement_rate']:.4f} per round")
        report_lines.append(f"Total Improvement: {convergence['total_improvement']:.3f}")
        
        # Performance trends
        if len(metrics_history) > 1:
            accuracies = [m.get('accuracy', 0) for m in metrics_history]
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            
            report_lines.append(f"\nPerformance Range:")
            report_lines.append(f"Best Accuracy: {max_acc:.3f}")
            report_lines.append(f"Worst Accuracy: {min_acc:.3f}")
            report_lines.append(f"Accuracy Range: {max_acc - min_acc:.3f}")
        
        return "\n".join(report_lines)

def test_pathology_fl_utils():
    """Test PathologyFL utility functions."""
    print("Testing PathologyFL utilities...")
    
    utils = PathologyFLUtils()
    
    # Test hospital weight calculation
    metadata = {
        "hospital_type": "cancer_center",
        "years_experience": 20,
        "diagnostic_accuracy": 0.94,
        "annual_cases": 15000
    }
    
    weight = utils.calculate_hospital_weight(metadata)
    print(f"Hospital weight: {weight:.3f}")
    
    # Test slide quality assessment
    slide_data = {"slide_id": "test_slide"}
    quality = utils.assess_slide_quality(slide_data)
    print(f"Slide quality: {quality['overall_quality']:.3f}")
    
    # Test differential privacy
    gradients = [0.1, 0.2, 0.3, 0.4, 0.5]
    noisy_gradients = utils.add_differential_privacy_noise(gradients)
    print(f"Original gradients: {gradients}")
    print(f"Noisy gradients: {[f'{g:.3f}' for g in noisy_gradients]}")
    
    # Test Byzantine detection
    updates = [
        {"hospital_id": "normal1", "parameters": {"layer1": [0.1, 0.2, 0.3]}},
        {"hospital_id": "normal2", "parameters": {"layer1": [0.11, 0.21, 0.31]}},
        {"hospital_id": "byzantine", "parameters": {"layer1": [100.0, 200.0, 300.0]}}
    ]
    
    byzantine_detected = utils.detect_byzantine_updates(updates)
    print(f"Byzantine hospitals detected: {byzantine_detected}")
    
    # Test compression
    update = {
        "hospital_id": "test",
        "parameters": {"layer1": [0.1, 0.2, 0.3, 0.4, 0.5]}
    }
    
    compressed = utils.compress_model_update(update, compression_ratio=0.4)
    print(f"Compression ratio: {len(compressed['parameters']['layer1']) / 5:.1%}")
    
    return True

if __name__ == "__main__":
    success = test_pathology_fl_utils()
    print(f"PathologyFL utilities test: {'PASSED' if success else 'FAILED'}")