#!/usr/bin/env python3
"""Test IMR vs Simple Ensemble to determine if complexity is worth it."""

import random
import time
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import IMR directly
from imr.intelligent_medical_referee import IntelligentMedicalReferee

def simple_ensemble(fl_prob, dmi_prob, fl_confidence=0.5, dmi_confidence=0.5):
    """Simple weighted average ensemble."""
    total_weight = fl_confidence + dmi_confidence
    return (fl_prob * fl_confidence + dmi_prob * dmi_confidence) / total_weight

def generate_test_cases(num_cases=100):
    """Generate test cases with known ground truth."""
    cases = []
    
    for i in range(num_cases):
        # Generate ground truth
        ground_truth = random.uniform(0.0, 1.0)
        
        # FL prediction (good with large datasets, some noise)
        fl_noise = random.uniform(-0.15, 0.15)
        fl_prob = max(0.0, min(1.0, ground_truth + fl_noise))
        fl_confidence = random.uniform(0.6, 0.9)
        
        # DMI prediction (better with expertise, but can be overconfident)
        if ground_truth > 0.7:  # High cancer cases - experts are very good
            dmi_noise = random.uniform(-0.05, 0.05)
            dmi_confidence = random.uniform(0.8, 0.95)
        else:  # Low cancer cases - experts sometimes overdiagnose
            dmi_noise = random.uniform(-0.1, 0.2)
            dmi_confidence = random.uniform(0.6, 0.8)
            
        dmi_prob = max(0.0, min(1.0, ground_truth + dmi_noise))
        
        cases.append({
            "ground_truth": ground_truth,
            "fl_prob": fl_prob,
            "fl_confidence": fl_confidence,
            "dmi_prob": dmi_prob,
            "dmi_confidence": dmi_confidence,
            "disagreement": abs(fl_prob - dmi_prob)
        })
    
    return cases

def test_accuracy_comparison():
    """Compare accuracy of IMR vs Simple Ensemble."""
    print("Testing accuracy comparison...")
    
    imr = IntelligentMedicalReferee()
    test_cases = generate_test_cases(100)
    
    imr_errors = []
    simple_errors = []
    
    for case in test_cases:
        ground_truth = case["ground_truth"]
        
        # Simple ensemble prediction
        simple_pred = simple_ensemble(
            case["fl_prob"], case["dmi_prob"], 
            case["fl_confidence"], case["dmi_confidence"]
        )
        simple_error = abs(simple_pred - ground_truth)
        simple_errors.append(simple_error)
        
        # IMR prediction
        fl_result = {
            "cancer_probability": case["fl_prob"],
            "model_confidence": case["fl_confidence"],
            "contributing_hospitals": 50,
            "total_training_samples": 100000
        }
        
        dmi_result = {
            "consensus_probability": case["dmi_prob"],
            "contributing_experts": 5,
            "total_expertise_weight": 12.0,
            "consensus_strength": case["dmi_confidence"]
        }
        
        imr_result = imr.arbitrate_predictions(fl_result, dmi_result, {"case_id": f"test_{len(imr_errors)}"})
        imr_pred = imr_result["final_probability"]
        imr_error = abs(imr_pred - ground_truth)
        imr_errors.append(imr_error)
    
    # Calculate metrics
    imr_mae = sum(imr_errors) / len(imr_errors)
    simple_mae = sum(simple_errors) / len(simple_errors)
    
    print(f"  IMR Mean Absolute Error: {imr_mae:.4f}")
    print(f"  Simple Ensemble MAE: {simple_mae:.4f}")
    print(f"  IMR Improvement: {((simple_mae - imr_mae) / simple_mae * 100):+.1f}%")
    
    return imr_mae, simple_mae

def test_performance_comparison():
    """Compare performance of IMR vs Simple Ensemble."""
    print("Testing performance comparison...")
    
    imr = IntelligentMedicalReferee()
    
    # Simple ensemble timing
    start_time = time.time()
    for _ in range(1000):
        simple_ensemble(0.7, 0.8, 0.9, 0.85)
    simple_time = time.time() - start_time
    
    # IMR timing
    fl_result = {
        "cancer_probability": 0.7,
        "model_confidence": 0.9,
        "contributing_hospitals": 50
    }
    dmi_result = {
        "consensus_probability": 0.8,
        "contributing_experts": 5,
        "consensus_strength": 0.85
    }
    
    start_time = time.time()
    for i in range(1000):
        imr.arbitrate_predictions(fl_result, dmi_result, {"case_id": f"perf_{i}"})
    imr_time = time.time() - start_time
    
    print(f"  Simple Ensemble: {simple_time:.4f}s (1000 predictions)")
    print(f"  IMR: {imr_time:.4f}s (1000 predictions)")
    print(f"  Performance overhead: {imr_time/simple_time:.1f}x slower")
    
    return simple_time, imr_time

def test_disagreement_handling():
    """Test how well each method handles disagreement cases."""
    print("Testing disagreement handling...")
    
    imr = IntelligentMedicalReferee()
    
    # Create cases with high disagreement
    disagreement_cases = [
        # FL says low, DMI says high (expert catches something FL missed)
        {"ground_truth": 0.85, "fl_prob": 0.3, "dmi_prob": 0.9},
        # FL says high, DMI says low (FL overfitting, expert knows better)
        {"ground_truth": 0.2, "fl_prob": 0.8, "dmi_prob": 0.25},
        # Both wrong in different directions
        {"ground_truth": 0.5, "fl_prob": 0.2, "dmi_prob": 0.9},
    ]
    
    imr_better = 0
    simple_better = 0
    
    for case in disagreement_cases:
        ground_truth = case["ground_truth"]
        
        # Simple ensemble (just averages)
        simple_pred = (case["fl_prob"] + case["dmi_prob"]) / 2
        simple_error = abs(simple_pred - ground_truth)
        
        # IMR (intelligent reasoning)
        fl_result = {"cancer_probability": case["fl_prob"], "model_confidence": 0.8}
        dmi_result = {"consensus_probability": case["dmi_prob"], "consensus_strength": 0.9}
        
        imr_result = imr.arbitrate_predictions(fl_result, dmi_result, {"case_id": "disagree"})
        imr_pred = imr_result["final_probability"]
        imr_error = abs(imr_pred - ground_truth)
        
        print(f"  Case: GT={ground_truth:.1f}, FL={case['fl_prob']:.1f}, DMI={case['dmi_prob']:.1f}")
        print(f"    Simple: {simple_pred:.2f} (error: {simple_error:.3f})")
        print(f"    IMR: {imr_pred:.2f} (error: {imr_error:.3f})")
        
        if imr_error < simple_error:
            imr_better += 1
        else:
            simple_better += 1
    
    print(f"  IMR better: {imr_better}/{len(disagreement_cases)}")
    print(f"  Simple better: {simple_better}/{len(disagreement_cases)}")
    
    return imr_better > simple_better

def run_comparison_tests():
    """Run all comparison tests to determine if IMR complexity is worth it."""
    print("🔬 IMR vs Simple Ensemble Comparison")
    print("=" * 60)
    
    # Test accuracy
    imr_mae, simple_mae = test_accuracy_comparison()
    accuracy_improvement = (simple_mae - imr_mae) / simple_mae * 100
    print()
    
    # Test performance
    simple_time, imr_time = test_performance_comparison()
    performance_overhead = imr_time / simple_time
    print()
    
    # Test disagreement handling
    imr_handles_disagreement_better = test_disagreement_handling()
    print()
    
    # Final verdict
    print("=" * 60)
    print("📊 FINAL VERDICT")
    print("=" * 60)
    
    print(f"Accuracy: IMR is {accuracy_improvement:+.1f}% better")
    print(f"Performance: IMR is {performance_overhead:.1f}x slower")
    print(f"Disagreement handling: {'IMR better' if imr_handles_disagreement_better else 'Simple better'}")
    
    # Decision logic
    if accuracy_improvement > 5 and imr_handles_disagreement_better:
        verdict = "✅ IMR WORTH THE COMPLEXITY"
        reason = "Significant accuracy improvement + better disagreement handling"
    elif accuracy_improvement > 10:
        verdict = "✅ IMR WORTH THE COMPLEXITY"
        reason = "Major accuracy improvement outweighs complexity"
    elif performance_overhead > 10:
        verdict = "❌ STICK WITH SIMPLE ENSEMBLE"
        reason = "Too much performance overhead for minimal benefit"
    elif accuracy_improvement < 1:
        verdict = "❌ STICK WITH SIMPLE ENSEMBLE"
        reason = "No meaningful accuracy improvement"
    else:
        verdict = "🤔 BORDERLINE - DEPENDS ON USE CASE"
        reason = "Modest improvements, evaluate based on specific needs"
    
    print(f"\n{verdict}")
    print(f"Reason: {reason}")
    
    return verdict.startswith("✅")

if __name__ == "__main__":
    success = run_comparison_tests()
    exit(0 if success else 1)