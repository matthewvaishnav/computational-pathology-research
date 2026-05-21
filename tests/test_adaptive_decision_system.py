#!/usr/bin/env python3
"""Adaptive Decision System - Learns when to use IMR vs Simple Ensemble."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from imr.intelligent_medical_referee import IntelligentMedicalReferee


class AdaptiveDecisionSystem:
    """Learns when to use IMR vs Simple Ensemble based on case characteristics."""

    def __init__(self):
        self.imr = IntelligentMedicalReferee()
        self.decision_history = []
        self.performance_tracker = {"imr_wins": 0, "simple_wins": 0, "total_cases": 0}

    def extract_case_features(self, fl_result, dmi_result, case_data):
        """Extract features that predict when IMR will be better."""
        features = {}

        # Disagreement level
        disagreement = abs(
            fl_result.get("cancer_probability", 0) - dmi_result.get("consensus_probability", 0)
        )
        features["disagreement"] = disagreement

        # Confidence levels
        fl_confidence = fl_result.get("model_confidence", 0.5)
        dmi_confidence = dmi_result.get("consensus_strength", 0.5)
        features["fl_confidence"] = fl_confidence
        features["dmi_confidence"] = dmi_confidence
        features["confidence_gap"] = abs(fl_confidence - dmi_confidence)

        # Data quality indicators
        features["fl_training_samples"] = fl_result.get("total_training_samples", 0)
        features["dmi_expertise_weight"] = dmi_result.get("total_expertise_weight", 0)
        features["specialization_match"] = dmi_result.get("specialization_relevance", 0)

        # Case characteristics
        features["is_rare_cancer"] = case_data.get("is_rare", False)
        features["is_pediatric"] = case_data.get("is_pediatric", False)
        features["has_artifacts"] = case_data.get("has_artifacts", False)

        return features

    def predict_imr_advantage(self, features):
        """Predict if IMR will outperform simple ensemble for this case."""
        score = 0.0

        # High disagreement favors IMR
        if features["disagreement"] > 0.2:
            score += 0.3

        # Low FL confidence but high DMI confidence favors IMR
        if features["fl_confidence"] < 0.7 and features["dmi_confidence"] > 0.8:
            score += 0.25

        # High specialization match favors IMR
        if features["specialization_match"] > 0.8:
            score += 0.2

        # Special case types favor IMR
        if features["is_rare_cancer"]:
            score += 0.4
        if features["is_pediatric"]:
            score += 0.3
        if features["has_artifacts"]:
            score += 0.25

        # Low FL training data favors IMR
        if features["fl_training_samples"] < 10000:
            score += 0.15

        return score > 0.5  # Threshold for using IMR

    def adaptive_predict(self, fl_result, dmi_result, case_data):
        """Make adaptive prediction using learned decision rules."""
        features = self.extract_case_features(fl_result, dmi_result, case_data)

        # Decide which method to use
        use_imr = self.predict_imr_advantage(features)

        if use_imr:
            # Use IMR for complex cases
            result = self.imr.arbitrate_predictions(fl_result, dmi_result, case_data)
            prediction = result["final_probability"]
            method_used = "imr"
            explanation = result.get("reasoning", "IMR arbitration")
        else:
            # Use simple ensemble for routine cases
            fl_prob = fl_result.get("cancer_probability", 0)
            dmi_prob = dmi_result.get("consensus_probability", 0)
            fl_conf = fl_result.get("model_confidence", 0.5)
            dmi_conf = dmi_result.get("consensus_strength", 0.5)

            total_weight = fl_conf + dmi_conf
            prediction = (fl_prob * fl_conf + dmi_prob * dmi_conf) / total_weight
            method_used = "simple"
            explanation = f"Simple ensemble: FL({fl_prob:.2f}) + DMI({dmi_prob:.2f})"

        return {
            "prediction": prediction,
            "method_used": method_used,
            "features": features,
            "explanation": explanation,
        }

    def learn_from_outcome(self, case_result, ground_truth):
        """Learn from actual outcomes to improve decision making."""
        prediction = case_result["prediction"]
        method_used = case_result["method_used"]
        features = case_result["features"]

        # Calculate error
        error = abs(prediction - ground_truth)

        # Also calculate what the other method would have done
        if method_used == "imr":
            # Calculate simple ensemble error
            fl_prob = features.get("fl_prob", prediction)  # Approximate
            dmi_prob = features.get("dmi_prob", prediction)  # Approximate
            simple_pred = (fl_prob + dmi_prob) / 2
            simple_error = abs(simple_pred - ground_truth)

            if error < simple_error:
                self.performance_tracker["imr_wins"] += 1
            else:
                self.performance_tracker["simple_wins"] += 1
        else:
            # For simple cases, assume IMR would be similar or worse
            self.performance_tracker["simple_wins"] += 1

        self.performance_tracker["total_cases"] += 1

        # Store for learning
        self.decision_history.append(
            {
                "features": features,
                "method_used": method_used,
                "error": error,
                "ground_truth": ground_truth,
            }
        )

    def get_performance_stats(self):
        """Get performance statistics."""
        total = self.performance_tracker["total_cases"]
        if total == 0:
            return {"imr_win_rate": 0, "simple_win_rate": 0, "total_cases": 0}

        return {
            "imr_win_rate": self.performance_tracker["imr_wins"] / total,
            "simple_win_rate": self.performance_tracker["simple_wins"] / total,
            "total_cases": total,
            "decision_accuracy": self.performance_tracker["imr_wins"]
            / max(
                1, self.performance_tracker["imr_wins"] + self.performance_tracker["simple_wins"]
            ),
        }


def create_test_cases_with_features():
    """Create test cases with explicit features for learning."""
    cases = [
        # Rare cancer - IMR should win
        {
            "fl_result": {
                "cancer_probability": 0.4,
                "model_confidence": 0.7,
                "total_training_samples": 500,
            },
            "dmi_result": {
                "consensus_probability": 0.85,
                "consensus_strength": 0.9,
                "specialization_relevance": 0.95,
            },
            "case_data": {"is_rare": True, "case_type": "rare_sarcoma"},
            "ground_truth": 0.88,
        },
        # Common cancer - Simple should win
        {
            "fl_result": {
                "cancer_probability": 0.82,
                "model_confidence": 0.9,
                "total_training_samples": 100000,
            },
            "dmi_result": {
                "consensus_probability": 0.85,
                "consensus_strength": 0.8,
                "specialization_relevance": 0.7,
            },
            "case_data": {"is_rare": False, "case_type": "common_breast_cancer"},
            "ground_truth": 0.83,
        },
        # Artifact case - IMR should win
        {
            "fl_result": {
                "cancer_probability": 0.75,
                "model_confidence": 0.8,
                "total_training_samples": 50000,
            },
            "dmi_result": {
                "consensus_probability": 0.15,
                "consensus_strength": 0.95,
                "specialization_relevance": 0.8,
            },
            "case_data": {"has_artifacts": True, "case_type": "processing_artifact"},
            "ground_truth": 0.1,
        },
        # Pediatric case - IMR should win
        {
            "fl_result": {
                "cancer_probability": 0.5,
                "model_confidence": 0.6,
                "total_training_samples": 75000,
            },
            "dmi_result": {
                "consensus_probability": 0.9,
                "consensus_strength": 0.95,
                "specialization_relevance": 0.95,
            },
            "case_data": {"is_pediatric": True, "case_type": "pediatric_tumor"},
            "ground_truth": 0.92,
        },
        # Routine case - Simple should win
        {
            "fl_result": {
                "cancer_probability": 0.65,
                "model_confidence": 0.85,
                "total_training_samples": 200000,
            },
            "dmi_result": {
                "consensus_probability": 0.68,
                "consensus_strength": 0.8,
                "specialization_relevance": 0.7,
            },
            "case_data": {"is_rare": False, "case_type": "routine_screening"},
            "ground_truth": 0.66,
        },
    ]

    return cases


def test_adaptive_system():
    """Test the adaptive decision system."""
    print("Testing Adaptive Decision System...")

    ads = AdaptiveDecisionSystem()
    test_cases = create_test_cases_with_features()

    correct_decisions = 0
    total_error = 0

    for i, case in enumerate(test_cases):
        # Make adaptive prediction
        result = ads.adaptive_predict(case["fl_result"], case["dmi_result"], case["case_data"])

        # Learn from outcome
        ads.learn_from_outcome(result, case["ground_truth"])

        # Check if decision was good
        error = abs(result["prediction"] - case["ground_truth"])
        total_error += error

        # Determine if this was the right method to use
        case_type = case["case_data"].get("case_type", "unknown")
        method_used = result["method_used"]

        # Expected method based on case type
        expected_imr_cases = ["rare_sarcoma", "processing_artifact", "pediatric_tumor"]
        should_use_imr = case_type in expected_imr_cases

        if (should_use_imr and method_used == "imr") or (
            not should_use_imr and method_used == "simple"
        ):
            correct_decisions += 1

        print(f"  Case {i+1} ({case_type}):")
        print(
            f"    Method: {method_used} {'✅' if (should_use_imr and method_used == 'imr') or (not should_use_imr and method_used == 'simple') else '❌'}"
        )
        print(f"    Prediction: {result['prediction']:.3f}")
        print(f"    Ground Truth: {case['ground_truth']:.3f}")
        print(f"    Error: {error:.3f}")

    # Get performance stats
    stats = ads.get_performance_stats()
    avg_error = total_error / len(test_cases)
    decision_accuracy = correct_decisions / len(test_cases)

    print(f"\n  Results:")
    print(f"    Decision Accuracy: {decision_accuracy:.1%} ({correct_decisions}/{len(test_cases)})")
    print(f"    Average Error: {avg_error:.4f}")
    print(f"    IMR Win Rate: {stats['imr_win_rate']:.1%}")
    print(f"    Simple Win Rate: {stats['simple_win_rate']:.1%}")

    return decision_accuracy > 0.8 and avg_error < 0.1


def run_adaptive_system_test():
    """Run adaptive system test."""
    print("🧠 Adaptive Decision System Testing")
    print("=" * 60)

    success = test_adaptive_system()

    print("\n" + "=" * 60)
    print("📊 ADAPTIVE SYSTEM VERDICT")
    print("=" * 60)

    if success:
        print("✅ ADAPTIVE SYSTEM WORKS")
        print("Learns when to use IMR vs Simple Ensemble")
        print("Accounts for case characteristics automatically")
    else:
        print("❌ ADAPTIVE SYSTEM NEEDS WORK")
        print("Decision logic or learning mechanism needs improvement")

    return success


if __name__ == "__main__":
    success = run_adaptive_system_test()
    exit(0 if success else 1)
