#!/usr/bin/env python3
"""Realistic medical scenario test for IMR vs Simple Ensemble."""

import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from imr.intelligent_medical_referee import IntelligentMedicalReferee

def create_realistic_medical_scenarios():
    """Create realistic medical scenarios where expertise matters."""
    
    scenarios = [
        # Rare cancer type - experts know, FL doesn't have enough data
        {
            "name": "Rare Angiosarcoma",
            "ground_truth": 0.85,
            "fl_prob": 0.35,  # FL trained on common cancers, misses rare type
            "fl_confidence": 0.7,
            "fl_training_samples": 50,  # Very few rare cancer samples
            "dmi_prob": 0.90,  # Specialist recognizes rare pattern
            "dmi_confidence": 0.95,
            "expert_specialization": 0.95,  # Perfect match
            "clinical_indicators": {"rare_vascular_pattern": 0.9, "spindle_cells": 0.8}
        },
        
        # Inflammatory condition mimicking cancer - expert catches it
        {
            "name": "Inflammatory Pseudotumor",
            "ground_truth": 0.15,  # Not cancer, just inflammation
            "fl_prob": 0.80,  # FL sees "tumor-like" features
            "fl_confidence": 0.8,
            "fl_training_samples": 100000,
            "dmi_prob": 0.20,  # Expert recognizes inflammatory pattern
            "dmi_confidence": 0.90,
            "expert_specialization": 0.85,
            "clinical_indicators": {"inflammatory_cells": 0.9, "necrosis": 0.1}
        },
        
        # Borderline case where both systems struggle
        {
            "name": "Atypical Ductal Hyperplasia",
            "ground_truth": 0.45,  # Borderline malignant potential
            "fl_prob": 0.60,
            "fl_confidence": 0.6,  # FL uncertain
            "fl_training_samples": 75000,
            "dmi_prob": 0.40,
            "dmi_confidence": 0.65,  # Experts also uncertain
            "expert_specialization": 0.90,
            "clinical_indicators": {"cellular_atypia": 0.6, "architectural_distortion": 0.5}
        },
        
        # Clear cancer case - both should agree
        {
            "name": "Invasive Ductal Carcinoma",
            "ground_truth": 0.95,
            "fl_prob": 0.92,
            "fl_confidence": 0.95,
            "fl_training_samples": 200000,
            "dmi_prob": 0.94,
            "dmi_confidence": 0.98,
            "expert_specialization": 0.95,
            "clinical_indicators": {"invasion": 0.95, "pleomorphism": 0.9}
        },
        
        # Artifact/technical issue - expert catches, FL fooled
        {
            "name": "Tissue Folding Artifact",
            "ground_truth": 0.05,  # Not cancer, just artifact
            "fl_prob": 0.75,  # FL sees "suspicious" overlapping tissue
            "fl_confidence": 0.7,
            "fl_training_samples": 150000,
            "dmi_prob": 0.10,  # Expert recognizes artifact
            "dmi_confidence": 0.95,
            "expert_specialization": 0.80,
            "clinical_indicators": {"tissue_folding": 0.9, "processing_artifact": 0.8}
        }
    ]
    
    return scenarios

def test_realistic_scenarios():
    """Test IMR vs Simple Ensemble on realistic medical scenarios."""
    print("Testing realistic medical scenarios...")
    
    imr = IntelligentMedicalReferee()
    scenarios = create_realistic_medical_scenarios()
    
    imr_errors = []
    simple_errors = []
    imr_better_cases = []
    
    for scenario in scenarios:
        ground_truth = scenario["ground_truth"]
        
        # Simple ensemble
        fl_weight = scenario["fl_confidence"]
        dmi_weight = scenario["dmi_confidence"]
        total_weight = fl_weight + dmi_weight
        
        simple_pred = (scenario["fl_prob"] * fl_weight + scenario["dmi_prob"] * dmi_weight) / total_weight
        simple_error = abs(simple_pred - ground_truth)
        simple_errors.append(simple_error)
        
        # IMR prediction
        fl_result = {
            "cancer_probability": scenario["fl_prob"],
            "model_confidence": scenario["fl_confidence"],
            "contributing_hospitals": 50,
            "total_training_samples": scenario["fl_training_samples"],
            "feature_weights": scenario.get("clinical_indicators", {})
        }
        
        dmi_result = {
            "consensus_probability": scenario["dmi_prob"],
            "contributing_experts": 5,
            "total_expertise_weight": 15.0,
            "consensus_strength": scenario["dmi_confidence"],
            "specialization_relevance": scenario["expert_specialization"],
            "clinical_indicators": scenario.get("clinical_indicators", {})
        }
        
        imr_result = imr.arbitrate_predictions(fl_result, dmi_result, {"case_id": scenario["name"]})
        imr_pred = imr_result["final_probability"]
        imr_error = abs(imr_pred - ground_truth)
        imr_errors.append(imr_error)
        
        # Track which method is better for this case
        if imr_error < simple_error:
            imr_better_cases.append(scenario["name"])
        
        print(f"\n  {scenario['name']}:")
        print(f"    Ground Truth: {ground_truth:.2f}")
        print(f"    FL: {scenario['fl_prob']:.2f} (conf: {scenario['fl_confidence']:.2f})")
        print(f"    DMI: {scenario['dmi_prob']:.2f} (conf: {scenario['dmi_confidence']:.2f})")
        print(f"    Simple: {simple_pred:.2f} (error: {simple_error:.3f})")
        print(f"    IMR: {imr_pred:.2f} (error: {imr_error:.3f}) {'✅' if imr_error < simple_error else '❌'}")
        print(f"    Decision: {imr_result['decision_type']}")
    
    # Calculate overall metrics
    imr_mae = sum(imr_errors) / len(imr_errors)
    simple_mae = sum(simple_errors) / len(simple_errors)
    improvement = (simple_mae - imr_mae) / simple_mae * 100
    
    print(f"\n  Overall Results:")
    print(f"    IMR MAE: {imr_mae:.4f}")
    print(f"    Simple MAE: {simple_mae:.4f}")
    print(f"    IMR Improvement: {improvement:+.1f}%")
    print(f"    IMR Better Cases: {len(imr_better_cases)}/{len(scenarios)}")
    print(f"    Cases where IMR won: {imr_better_cases}")
    
    return imr_mae < simple_mae, improvement

def test_high_stakes_scenarios():
    """Test scenarios where being wrong has high cost."""
    print("\nTesting high-stakes scenarios...")
    
    imr = IntelligentMedicalReferee()
    
    # High-stakes scenarios where expert knowledge is critical
    high_stakes = [
        # Pediatric case - very different from adult training data
        {
            "name": "Pediatric Rhabdomyosarcoma",
            "ground_truth": 0.90,
            "fl_prob": 0.45,  # Adult-trained model struggles
            "dmi_prob": 0.88,  # Pediatric specialist knows
            "cost_of_miss": 100,  # Very high cost if missed
        },
        
        # Metastatic vs primary - critical for treatment
        {
            "name": "Metastatic Breast Cancer to Brain",
            "ground_truth": 0.85,  # Is metastatic
            "fl_prob": 0.55,  # Sees cancer but misses metastatic nature
            "dmi_prob": 0.82,  # Expert recognizes metastatic pattern
            "cost_of_miss": 80,
        },
        
        # Benign mimic - unnecessary treatment if wrong
        {
            "name": "Sclerosing Adenosis",
            "ground_truth": 0.10,  # Benign but looks suspicious
            "fl_prob": 0.70,  # FL fooled by appearance
            "dmi_prob": 0.15,  # Expert knows the mimic
            "cost_of_miss": 60,  # Unnecessary surgery/chemo
        }
    ]
    
    total_cost_simple = 0
    total_cost_imr = 0
    
    for case in high_stakes:
        # Simple ensemble
        simple_pred = (case["fl_prob"] + case["dmi_prob"]) / 2
        simple_error = abs(simple_pred - case["ground_truth"])
        simple_cost = simple_error * case["cost_of_miss"]
        
        # IMR
        fl_result = {"cancer_probability": case["fl_prob"], "model_confidence": 0.8}
        dmi_result = {"consensus_probability": case["dmi_prob"], "consensus_strength": 0.9}
        
        imr_result = imr.arbitrate_predictions(fl_result, dmi_result, {"case_id": case["name"]})
        imr_pred = imr_result["final_probability"]
        imr_error = abs(imr_pred - case["ground_truth"])
        imr_cost = imr_error * case["cost_of_miss"]
        
        total_cost_simple += simple_cost
        total_cost_imr += imr_cost
        
        print(f"  {case['name']}:")
        print(f"    Simple cost: {simple_cost:.1f}")
        print(f"    IMR cost: {imr_cost:.1f} {'✅' if imr_cost < simple_cost else '❌'}")
    
    print(f"\n  Total Cost Comparison:")
    print(f"    Simple Ensemble: {total_cost_simple:.1f}")
    print(f"    IMR: {total_cost_imr:.1f}")
    print(f"    Cost Reduction: {((total_cost_simple - total_cost_imr) / total_cost_simple * 100):+.1f}%")
    
    return total_cost_imr < total_cost_simple

def run_realistic_comparison():
    """Run realistic medical scenario comparison."""
    print("🏥 Realistic Medical Scenario Testing")
    print("=" * 60)
    
    # Test realistic scenarios
    imr_better_accuracy, improvement = test_realistic_scenarios()
    
    # Test high-stakes scenarios
    imr_better_cost = test_high_stakes_scenarios()
    
    print("\n" + "=" * 60)
    print("📊 REALISTIC SCENARIO VERDICT")
    print("=" * 60)
    
    if imr_better_accuracy and improvement > 5:
        verdict = "✅ IMR JUSTIFIED FOR MEDICAL USE"
        reason = f"IMR {improvement:+.1f}% better on realistic medical scenarios"
    elif imr_better_cost:
        verdict = "✅ IMR JUSTIFIED FOR HIGH-STAKES CASES"
        reason = "IMR reduces cost of medical errors significantly"
    elif imr_better_accuracy:
        verdict = "🤔 IMR USEFUL FOR SPECIFIC MEDICAL CASES"
        reason = "IMR better on realistic scenarios but improvement modest"
    else:
        verdict = "❌ SIMPLE ENSEMBLE STILL BETTER"
        reason = "Even realistic scenarios don't justify IMR complexity"
    
    print(f"{verdict}")
    print(f"Reason: {reason}")
    
    return verdict.startswith("✅")

if __name__ == "__main__":
    success = run_realistic_comparison()
    exit(0 if success else 1)