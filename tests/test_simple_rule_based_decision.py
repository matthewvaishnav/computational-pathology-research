#!/usr/bin/env python3
"""Simple Rule-Based Decision System - Transparent and reliable routing."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from imr.intelligent_medical_referee import IntelligentMedicalReferee

class SimpleRuleBasedDecision:
    """Simple, transparent rules for when to use IMR vs Simple Ensemble."""
    
    def __init__(self):
        self.imr = IntelligentMedicalReferee()
        self.decision_log = []
        
    def should_use_imr(self, fl_result, dmi_result, case_data):
        """Simple, transparent rules for IMR usage."""
        
        # Rule 1: High disagreement (>20% difference)
        disagreement = abs(fl_result.get("cancer_probability", 0) - 
                          dmi_result.get("consensus_probability", 0))
        if disagreement > 0.2:
            return True, "high_disagreement"
        
        # Rule 2: Low FL confidence but high DMI confidence
        fl_conf = fl_result.get("model_confidence", 0.5)
        dmi_conf = dmi_result.get("consensus_strength", 0.5)
        if fl_conf < 0.7 and dmi_conf > 0.85:
            return True, "expert_confidence_advantage"
        
        # Rule 3: Explicit high-stakes markers
        if case_data.get("is_rare", False):
            return True, "rare_cancer"
        if case_data.get("is_pediatric", False):
            return True, "pediatric_case"
        if case_data.get("has_artifacts", False):
            return True, "artifact_suspected"
        
        # Rule 4: Very low FL training data
        if fl_result.get("total_training_samples", float('inf')) < 5000:
            return True, "insufficient_training_data"
        
        # Default: use simple ensemble
        return False, "routine_case"
    
    def predict(self, fl_result, dmi_result, case_data):
        """Make prediction with transparent decision logic."""
        
        use_imr, reason = self.should_use_imr(fl_result, dmi_result, case_data)
        
        if use_imr:
            # Use IMR
            imr_result = self.imr.arbitrate_predictions(fl_result, dmi_result, case_data)
            prediction = imr_result["final_probability"]
            method = "imr"
            explanation = f"Used IMR because: {reason}"
        else:
            # Use simple ensemble
            fl_prob = fl_result.get("cancer_probability", 0)
            dmi_prob = dmi_result.get("consensus_probability", 0)
            fl_conf = fl_result.get("model_confidence", 0.5)
            dmi_conf = dmi_result.get("consensus_strength", 0.5)
            
            total_weight = fl_conf + dmi_conf
            prediction = (fl_prob * fl_conf + dmi_prob * dmi_conf) / total_weight
            method = "simple"
            explanation = f"Used Simple Ensemble because: {reason}"
        
        # Log decision for transparency
        decision_record = {
            "method": method,
            "reason": reason,
            "prediction": prediction,
            "fl_prob": fl_result.get("cancer_probability", 0),
            "dmi_prob": dmi_result.get("consensus_probability", 0),
            "disagreement": abs(fl_result.get("cancer_probability", 0) - 
                              dmi_result.get("consensus_probability", 0))
        }
        self.decision_log.append(decision_record)
        
        return {
            "prediction": prediction,
            "method": method,
            "reason": reason,
            "explanation": explanation,
            "decision_record": decision_record
        }
    
    def get_decision_stats(self):
        """Get transparent statistics about decisions made."""
        if not self.decision_log:
            return {}
        
        total = len(self.decision_log)
        imr_count = sum(1 for d in self.decision_log if d["method"] == "imr")
        
        # Count reasons
        reason_counts = {}
        for decision in self.decision_log:
            reason = decision["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            "total_decisions": total,
            "imr_usage_rate": imr_count / total,
            "simple_usage_rate": (total - imr_count) / total,
            "reason_breakdown": reason_counts,
            "avg_disagreement_imr": sum(d["disagreement"] for d in self.decision_log if d["method"] == "imr") / max(1, imr_count),
            "avg_disagreement_simple": sum(d["disagreement"] for d in self.decision_log if d["method"] == "simple") / max(1, total - imr_count)
        }

def test_rule_based_system():
    """Test the simple rule-based system."""
    print("Testing Simple Rule-Based Decision System...")
    
    rbd = SimpleRuleBasedDecision()
    
    # Test cases with clear expected outcomes
    test_cases = [
        # High disagreement case - should use IMR
        {
            "name": "High Disagreement",
            "fl_result": {"cancer_probability": 0.3, "model_confidence": 0.8},
            "dmi_result": {"consensus_probability": 0.8, "consensus_strength": 0.9},
            "case_data": {},
            "expected_method": "imr",
            "expected_reason": "high_disagreement"
        },
        
        # Rare cancer - should use IMR
        {
            "name": "Rare Cancer",
            "fl_result": {"cancer_probability": 0.6, "model_confidence": 0.7},
            "dmi_result": {"consensus_probability": 0.65, "consensus_strength": 0.8},
            "case_data": {"is_rare": True},
            "expected_method": "imr",
            "expected_reason": "rare_cancer"
        },
        
        # Routine case - should use simple
        {
            "name": "Routine Case",
            "fl_result": {"cancer_probability": 0.7, "model_confidence": 0.85, "total_training_samples": 100000},
            "dmi_result": {"consensus_probability": 0.72, "consensus_strength": 0.8},
            "case_data": {},
            "expected_method": "simple",
            "expected_reason": "routine_case"
        },
        
        # Low FL confidence, high DMI confidence - should use IMR
        {
            "name": "Expert Advantage",
            "fl_result": {"cancer_probability": 0.5, "model_confidence": 0.6},
            "dmi_result": {"consensus_probability": 0.85, "consensus_strength": 0.9},
            "case_data": {},
            "expected_method": "imr",
            "expected_reason": "expert_confidence_advantage"
        },
        
        # Pediatric case - should use IMR
        {
            "name": "Pediatric Case",
            "fl_result": {"cancer_probability": 0.6, "model_confidence": 0.8},
            "dmi_result": {"consensus_probability": 0.65, "consensus_strength": 0.85},
            "case_data": {"is_pediatric": True},
            "expected_method": "imr",
            "expected_reason": "pediatric_case"
        }
    ]
    
    correct_decisions = 0
    
    for case in test_cases:
        result = rbd.predict(case["fl_result"], case["dmi_result"], case["case_data"])
        
        method_correct = result["method"] == case["expected_method"]
        reason_correct = result["reason"] == case["expected_reason"]
        
        if method_correct and reason_correct:
            correct_decisions += 1
        
        print(f"  {case['name']}:")
        print(f"    Expected: {case['expected_method']} ({case['expected_reason']})")
        print(f"    Actual: {result['method']} ({result['reason']}) {'✅' if method_correct and reason_correct else '❌'}")
        print(f"    Prediction: {result['prediction']:.3f}")
    
    # Get decision statistics
    stats = rbd.get_decision_stats()
    
    print(f"\n  Decision Statistics:")
    print(f"    Total decisions: {stats['total_decisions']}")
    print(f"    IMR usage rate: {stats['imr_usage_rate']:.1%}")
    print(f"    Simple usage rate: {stats['simple_usage_rate']:.1%}")
    print(f"    Reason breakdown: {stats['reason_breakdown']}")
    
    accuracy = correct_decisions / len(test_cases)
    print(f"    Rule accuracy: {accuracy:.1%} ({correct_decisions}/{len(test_cases)})")
    
    return accuracy == 1.0  # All rules should work correctly

def test_rule_transparency():
    """Test that rules are transparent and explainable."""
    print("\nTesting rule transparency...")
    
    rbd = SimpleRuleBasedDecision()
    
    # Test case
    fl_result = {"cancer_probability": 0.4, "model_confidence": 0.6}
    dmi_result = {"consensus_probability": 0.9, "consensus_strength": 0.95}
    case_data = {"is_rare": True}
    
    result = rbd.predict(fl_result, dmi_result, case_data)
    
    print(f"  Case: Rare cancer with expert confidence")
    print(f"  Decision: {result['method']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Explanation: {result['explanation']}")
    
    # Check that decision is explainable
    has_clear_reason = result['reason'] in ['rare_cancer', 'high_disagreement', 'expert_confidence_advantage', 'pediatric_case', 'artifact_suspected', 'routine_case']
    has_explanation = len(result['explanation']) > 10
    
    print(f"  Has clear reason: {has_clear_reason}")
    print(f"  Has explanation: {has_explanation}")
    
    return has_clear_reason and has_explanation

def run_rule_based_test():
    """Run rule-based system test."""
    print("📋 Simple Rule-Based Decision System Testing")
    print("=" * 60)
    
    # Test rule accuracy
    rules_work = test_rule_based_system()
    
    # Test transparency
    is_transparent = test_rule_transparency()
    
    print("\n" + "=" * 60)
    print("📊 RULE-BASED SYSTEM VERDICT")
    print("=" * 60)
    
    if rules_work and is_transparent:
        print("✅ SIMPLE RULES WORK PERFECTLY")
        print("Advantages:")
        print("  • 100% predictable behavior")
        print("  • Fully transparent decisions")
        print("  • Easy to debug and modify")
        print("  • No training data required")
        print("  • No black box complexity")
        print("  • Immediate deployment ready")
    else:
        print("❌ RULES NEED REFINEMENT")
    
    return rules_work and is_transparent

if __name__ == "__main__":
    success = run_rule_based_test()
    exit(0 if success else 1)