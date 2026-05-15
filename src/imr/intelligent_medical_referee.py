#!/usr/bin/env python3
"""Intelligent Medical Referee (IMR) - Arbitrates between FL and DMI predictions."""

import time
from typing import Any, Dict, List, Tuple


class IntelligentMedicalReferee:
    """IMR: Analyzes and arbitrates between FL and DMI predictions."""

    def __init__(self):
        self.decision_history = []
        self.confidence_threshold = 0.15  # 15% difference triggers deep analysis

    def arbitrate_predictions(self, fl_result: Dict, dmi_result: Dict, case_data: Dict) -> Dict:
        """Arbitrate between FL and DMI predictions with reasoning."""

        # Extract key predictions
        fl_prob = fl_result.get("cancer_probability", 0.0)
        dmi_prob = dmi_result.get("consensus_probability", 0.0)

        # Calculate disagreement
        disagreement = abs(fl_prob - dmi_prob)

        # Analyze reasoning from both sides
        fl_reasoning = self._analyze_fl_reasoning(fl_result)
        dmi_reasoning = self._analyze_dmi_reasoning(dmi_result)

        # Cross-validate evidence
        evidence_alignment = self._cross_validate_evidence(fl_reasoning, dmi_reasoning, case_data)

        # Make referee decision
        if disagreement < self.confidence_threshold:
            # Close agreement - simple average
            final_prob = (fl_prob + dmi_prob) / 2
            decision_type = "consensus"
            confidence = 0.9
        else:
            # Significant disagreement - deep analysis
            referee_decision = self._deep_analysis_arbitration(
                fl_result, dmi_result, fl_reasoning, dmi_reasoning, evidence_alignment
            )
            final_prob = referee_decision["probability"]
            decision_type = referee_decision["decision_type"]
            confidence = referee_decision["confidence"]

        # Generate explanation
        explanation = self._generate_explanation(
            fl_prob,
            dmi_prob,
            final_prob,
            decision_type,
            fl_reasoning,
            dmi_reasoning,
            evidence_alignment,
        )

        result = {
            "final_probability": final_prob,
            "fl_probability": fl_prob,
            "dmi_probability": dmi_prob,
            "disagreement_level": disagreement,
            "decision_type": decision_type,
            "confidence": confidence,
            "reasoning": explanation,
            "evidence_alignment": evidence_alignment,
        }

        self.decision_history.append(result)
        return result

    def _analyze_fl_reasoning(self, fl_result: Dict) -> Dict:
        """Analyze FL model reasoning and evidence."""
        return {
            "data_sources": fl_result.get("contributing_hospitals", 0),
            "model_confidence": fl_result.get("model_confidence", 0.5),
            "feature_importance": fl_result.get("feature_weights", {}),
            "training_data_size": fl_result.get("total_training_samples", 0),
            "cross_validation_score": fl_result.get("cv_score", 0.0),
        }

    def _analyze_dmi_reasoning(self, dmi_result: Dict) -> Dict:
        """Analyze DMI expert reasoning and evidence."""
        return {
            "expert_count": dmi_result.get("contributing_experts", 0),
            "expertise_weight": dmi_result.get("total_expertise_weight", 0.0),
            "consensus_strength": dmi_result.get("consensus_strength", 0.0),
            "specialization_match": dmi_result.get("specialization_relevance", 0.0),
            "clinical_evidence": dmi_result.get("clinical_indicators", {}),
        }

    def _cross_validate_evidence(
        self, fl_reasoning: Dict, dmi_reasoning: Dict, case_data: Dict
    ) -> Dict:
        """Cross-validate evidence between FL and DMI."""
        alignment_score = 0.0
        evidence_points = []

        # Check if FL features align with clinical indicators
        fl_features = fl_reasoning.get("feature_importance", {})
        clinical_evidence = dmi_reasoning.get("clinical_evidence", {})

        # Feature alignment analysis
        common_indicators = set(fl_features.keys()) & set(clinical_evidence.keys())
        if common_indicators:
            feature_alignment = 0.0
            for indicator in common_indicators:
                fl_weight = fl_features[indicator]
                clinical_weight = clinical_evidence[indicator]

                # Check if both point in same direction
                if (fl_weight > 0.5 and clinical_weight > 0.5) or (
                    fl_weight < 0.5 and clinical_weight < 0.5
                ):
                    feature_alignment += 1.0
                    evidence_points.append(f"Both FL and experts agree on {indicator}")
                else:
                    evidence_points.append(f"FL and experts disagree on {indicator}")

            alignment_score = feature_alignment / len(common_indicators)

        # Data quality vs expertise quality
        data_quality = min(
            fl_reasoning.get("training_data_size", 0) / 100000, 1.0
        )  # Normalize to 100k samples
        expertise_quality = min(
            dmi_reasoning.get("expertise_weight", 0) / 10.0, 1.0
        )  # Normalize to weight 10

        return {
            "alignment_score": alignment_score,
            "evidence_points": evidence_points,
            "data_quality": data_quality,
            "expertise_quality": expertise_quality,
            "common_indicators": list(common_indicators),
        }

    def _deep_analysis_arbitration(
        self,
        fl_result: Dict,
        dmi_result: Dict,
        fl_reasoning: Dict,
        dmi_reasoning: Dict,
        evidence_alignment: Dict,
    ) -> Dict:
        """Perform deep analysis when FL and DMI significantly disagree."""

        fl_prob = fl_result.get("cancer_probability", 0.0)
        dmi_prob = dmi_result.get("consensus_probability", 0.0)

        # Scoring factors
        scores = {"fl_score": 0.0, "dmi_score": 0.0}

        # Factor 1: Evidence alignment
        alignment_score = evidence_alignment.get("alignment_score", 0.0)
        if alignment_score > 0.7:
            # High alignment - trust the average
            final_prob = (fl_prob + dmi_prob) / 2
            decision_type = "evidence_aligned"
            confidence = 0.8
        else:
            # Low alignment - need to pick a side

            # Factor 2: Data quality vs expertise quality
            data_quality = evidence_alignment.get("data_quality", 0.0)
            expertise_quality = evidence_alignment.get("expertise_quality", 0.0)

            scores["fl_score"] += data_quality * 0.4
            scores["dmi_score"] += expertise_quality * 0.4

            # Factor 3: Confidence levels
            fl_confidence = fl_reasoning.get("model_confidence", 0.5)
            dmi_consensus = dmi_reasoning.get("consensus_strength", 0.5)

            scores["fl_score"] += fl_confidence * 0.3
            scores["dmi_score"] += dmi_consensus * 0.3

            # Factor 4: Specialization relevance
            specialization = dmi_reasoning.get("specialization_match", 0.0)
            scores["dmi_score"] += specialization * 0.3

            # Make decision based on scores
            if scores["fl_score"] > scores["dmi_score"]:
                final_prob = fl_prob * 0.7 + dmi_prob * 0.3  # Lean toward FL
                decision_type = "fl_favored"
                confidence = 0.6 + (scores["fl_score"] - scores["dmi_score"]) * 0.2
            else:
                final_prob = dmi_prob * 0.7 + fl_prob * 0.3  # Lean toward DMI
                decision_type = "dmi_favored"
                confidence = 0.6 + (scores["dmi_score"] - scores["fl_score"]) * 0.2

        return {
            "probability": final_prob,
            "decision_type": decision_type,
            "confidence": min(confidence, 0.95),
            "scores": scores,
        }

    def _generate_explanation(
        self,
        fl_prob: float,
        dmi_prob: float,
        final_prob: float,
        decision_type: str,
        fl_reasoning: Dict,
        dmi_reasoning: Dict,
        evidence_alignment: Dict,
    ) -> str:
        """Generate human-readable explanation of referee decision."""

        explanation = []

        # Start with the disagreement analysis
        disagreement = abs(fl_prob - dmi_prob)
        explanation.append(f"FL Model: {fl_prob:.1%} cancer probability")
        explanation.append(f"Medical Experts: {dmi_prob:.1%} cancer probability")
        explanation.append(f"Disagreement: {disagreement:.1%}")

        # Explain the reasoning
        if decision_type == "consensus":
            explanation.append("✅ Close agreement - using average of both predictions")

        elif decision_type == "evidence_aligned":
            explanation.append("✅ Evidence from both sides aligns well - using balanced approach")

        elif decision_type == "fl_favored":
            explanation.append("⚖️ Favoring FL model because:")
            explanation.append(
                f"  • Strong data foundation ({fl_reasoning.get('training_data_size', 0):,} samples)"
            )
            explanation.append(
                f"  • High model confidence ({fl_reasoning.get('model_confidence', 0):.1%})"
            )

        elif decision_type == "dmi_favored":
            explanation.append("⚖️ Favoring medical experts because:")
            explanation.append(
                f"  • High expertise weight ({dmi_reasoning.get('expertise_weight', 0):.1f})"
            )
            explanation.append(
                f"  • Strong consensus ({dmi_reasoning.get('consensus_strength', 0):.1%})"
            )
            explanation.append(f"  • Relevant specialization match")

        # Add evidence analysis
        common_indicators = evidence_alignment.get("common_indicators", [])
        if common_indicators:
            explanation.append(f"📊 Both sides considered: {', '.join(common_indicators)}")

        explanation.append(f"🎯 Final Decision: {final_prob:.1%} cancer probability")

        return "\n".join(explanation)


def test_imr_consensus():
    """Test IMR with close agreement."""
    print("Testing IMR consensus decision...")

    imr = IntelligentMedicalReferee()

    # Close agreement case
    fl_result = {
        "cancer_probability": 0.85,
        "model_confidence": 0.9,
        "contributing_hospitals": 50,
        "feature_weights": {"tumor_size": 0.8, "irregular_borders": 0.7},
    }

    dmi_result = {
        "consensus_probability": 0.87,
        "contributing_experts": 5,
        "total_expertise_weight": 15.2,
        "clinical_indicators": {"tumor_size": 0.9, "irregular_borders": 0.8},
    }

    case_data = {"patient_id": "test_001"}

    result = imr.arbitrate_predictions(fl_result, dmi_result, case_data)

    print(f"  FL: {result['fl_probability']:.1%}")
    print(f"  DMI: {result['dmi_probability']:.1%}")
    print(f"  Final: {result['final_probability']:.1%}")
    print(f"  Decision: {result['decision_type']}")

    return 0.85 < result["final_probability"] < 0.87


def test_imr_disagreement():
    """Test IMR with significant disagreement."""
    print("Testing IMR disagreement arbitration...")

    imr = IntelligentMedicalReferee()

    # Significant disagreement case
    fl_result = {
        "cancer_probability": 0.65,
        "model_confidence": 0.7,
        "contributing_hospitals": 100,
        "total_training_samples": 500000,
    }

    dmi_result = {
        "consensus_probability": 0.92,
        "contributing_experts": 8,
        "total_expertise_weight": 25.5,
        "consensus_strength": 0.9,
        "specialization_relevance": 0.95,
    }

    case_data = {"patient_id": "test_002"}

    result = imr.arbitrate_predictions(fl_result, dmi_result, case_data)

    print(f"  FL: {result['fl_probability']:.1%}")
    print(f"  DMI: {result['dmi_probability']:.1%}")
    print(f"  Final: {result['final_probability']:.1%}")
    print(f"  Decision: {result['decision_type']}")
    print(f"  Reasoning: {result['reasoning'].split('🎯')[0]}...")

    return result["disagreement_level"] > 0.15


def run_imr_tests():
    """Run all IMR tests."""
    print("⚖️ Intelligent Medical Referee (IMR) Testing")
    print("=" * 60)

    tests = [
        ("Consensus Decision", test_imr_consensus),
        ("Disagreement Arbitration", test_imr_disagreement),
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

    print("=" * 60)
    print(f"IMR Tests: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🏆 Intelligent Medical Referee fully operational!")

    return passed == len(tests)


if __name__ == "__main__":
    success = run_imr_tests()
    exit(0 if success else 1)
