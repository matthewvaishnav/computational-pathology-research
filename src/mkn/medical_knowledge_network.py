#!/usr/bin/env python3
"""Medical Knowledge Network (MKN) - Collaborative diagnostic intelligence."""

import time
import random
from typing import Dict, List, Any

class MedicalKnowledgeNetwork:
    """MKN: Medical Knowledge Network for collaborative diagnostics."""
    
    def __init__(self):
        self.knowledge_graph = {}
        self.diagnostic_patterns = {}
        self.clinical_insights = {}
        self.expertise_network = {}
        
    def register_clinical_expert(self, expert_id: str, credentials: Dict):
        """Register clinical expert in the knowledge network."""
        expertise_score = self._calculate_clinical_expertise(credentials)
        
        self.expertise_network[expert_id] = {
            "credentials": credentials,
            "expertise_score": expertise_score,
            "diagnostic_contributions": 0,
            "accuracy_history": [],
            "specialties": credentials.get("clinical_specialties", [])
        }
        
        return expertise_score
    
    def _calculate_clinical_expertise(self, credentials: Dict) -> float:
        """Calculate clinical expertise score."""
        score = 1.0
        
        # Clinical experience years
        years = credentials.get("clinical_experience_years", 0)
        score += years * 0.05  # 5% per year
        
        # Fellowship training
        fellowships = credentials.get("fellowship_training", 0)
        score += fellowships * 0.3
        
        # Peer review ratings
        peer_rating = credentials.get("peer_review_rating", 3.0)  # 1-5 scale
        score *= (peer_rating / 3.0)
        
        # Case volume
        annual_cases = credentials.get("annual_case_volume", 1000)
        if annual_cases > 1000:
            import math
            score *= (1.0 + math.log10(annual_cases / 1000) * 0.2)
        
        return score
    
    def contribute_diagnostic_pattern(self, expert_id: str, pattern: Dict) -> Dict:
        """Contribute diagnostic pattern to knowledge network."""
        if expert_id not in self.expertise_network:
            return {"error": "Expert not registered"}
        
        pattern_id = f"{expert_id}_{int(time.time())}"
        expertise_score = self.expertise_network[expert_id]["expertise_score"]
        
        # Store pattern with expert weighting
        pattern_data = {
            "pattern": pattern,
            "expert_id": expert_id,
            "expertise_weight": expertise_score,
            "timestamp": time.time(),
            "validation_count": 0,
            "accuracy_score": 0.0
        }
        
        disease_type = pattern.get("disease_type", "unknown")
        if disease_type not in self.diagnostic_patterns:
            self.diagnostic_patterns[disease_type] = []
        
        self.diagnostic_patterns[disease_type].append(pattern_data)
        
        # Update expert contribution count
        self.expertise_network[expert_id]["diagnostic_contributions"] += 1
        
        return {"pattern_id": pattern_id, "expertise_weight": expertise_score}
    
    def query_diagnostic_knowledge(self, query: Dict) -> Dict:
        """Query diagnostic knowledge from the network."""
        disease_type = query.get("disease_type")
        symptoms = query.get("symptoms", [])
        
        if disease_type not in self.diagnostic_patterns:
            return {"error": "No patterns available for disease type"}
        
        patterns = self.diagnostic_patterns[disease_type]
        
        # Score patterns based on symptom matching and expert weight
        scored_patterns = []
        
        for pattern_data in patterns:
            pattern = pattern_data["pattern"]
            expert_weight = pattern_data["expertise_weight"]
            
            # Calculate symptom match score
            pattern_symptoms = pattern.get("symptoms", [])
            if pattern_symptoms:
                match_score = len(set(symptoms) & set(pattern_symptoms)) / len(set(symptoms) | set(pattern_symptoms))
            else:
                match_score = 0.0
            
            # Combined score: symptom match * expert weight
            combined_score = match_score * expert_weight
            
            scored_patterns.append({
                "pattern": pattern,
                "expert_id": pattern_data["expert_id"],
                "match_score": match_score,
                "expert_weight": expert_weight,
                "combined_score": combined_score
            })
        
        # Sort by combined score
        scored_patterns.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return {
            "disease_type": disease_type,
            "matched_patterns": scored_patterns[:5],  # Top 5 matches
            "total_patterns": len(patterns)
        }
    
    def validate_diagnostic_outcome(self, pattern_id: str, actual_outcome: Dict):
        """Validate diagnostic pattern with actual outcome."""
        # Find pattern and update accuracy
        for disease_type, patterns in self.diagnostic_patterns.items():
            for pattern_data in patterns:
                if pattern_data.get("pattern_id") == pattern_id:
                    # Calculate accuracy based on outcome match
                    predicted = pattern_data["pattern"].get("predicted_outcome", {})
                    
                    accuracy = self._calculate_outcome_accuracy(predicted, actual_outcome)
                    
                    # Update pattern accuracy
                    pattern_data["validation_count"] += 1
                    old_accuracy = pattern_data["accuracy_score"]
                    new_accuracy = (old_accuracy * (pattern_data["validation_count"] - 1) + accuracy) / pattern_data["validation_count"]
                    pattern_data["accuracy_score"] = new_accuracy
                    
                    # Update expert accuracy history
                    expert_id = pattern_data["expert_id"]
                    self.expertise_network[expert_id]["accuracy_history"].append(accuracy)
                    
                    return {"accuracy": accuracy, "updated_score": new_accuracy}
        
        return {"error": "Pattern not found"}
    
    def _calculate_outcome_accuracy(self, predicted: Dict, actual: Dict) -> float:
        """Calculate accuracy between predicted and actual outcomes."""
        if not predicted or not actual:
            return 0.0
        
        matches = 0
        total = 0
        
        for key in set(predicted.keys()) | set(actual.keys()):
            total += 1
            if key in predicted and key in actual:
                pred_val = predicted[key]
                actual_val = actual[key]
                
                if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                    # Numerical comparison with tolerance
                    if abs(pred_val - actual_val) / max(abs(actual_val), 1.0) < 0.1:  # 10% tolerance
                        matches += 1
                elif pred_val == actual_val:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def generate_consensus_diagnosis(self, case_data: Dict) -> Dict:
        """Generate consensus diagnosis from multiple expert patterns."""
        disease_type = case_data.get("suspected_disease")
        
        if disease_type not in self.diagnostic_patterns:
            return {"error": "No expert patterns available"}
        
        patterns = self.diagnostic_patterns[disease_type]
        
        # Weight patterns by expert expertise and historical accuracy
        weighted_predictions = {}
        total_weight = 0.0
        
        for pattern_data in patterns:
            expert_weight = pattern_data["expertise_weight"]
            accuracy_score = pattern_data.get("accuracy_score", 0.5)
            
            # Combined weight: expertise * accuracy
            combined_weight = expert_weight * (0.5 + accuracy_score)
            total_weight += combined_weight
            
            pattern = pattern_data["pattern"]
            predicted_outcome = pattern.get("predicted_outcome", {})
            
            for key, value in predicted_outcome.items():
                if key not in weighted_predictions:
                    weighted_predictions[key] = 0.0
                
                if isinstance(value, (int, float)):
                    weighted_predictions[key] += value * combined_weight
        
        # Normalize by total weight
        consensus = {}
        for key, weighted_sum in weighted_predictions.items():
            consensus[key] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            "consensus_diagnosis": consensus,
            "contributing_experts": len(patterns),
            "total_expertise_weight": total_weight,
            "confidence_score": min(total_weight / 10.0, 1.0)  # Normalize confidence
        }

def test_mkn_expert_registration():
    """Test MKN expert registration."""
    print("Testing MKN expert registration...")
    
    mkn = MedicalKnowledgeNetwork()
    
    experts = [
        ("dr_smith", {
            "clinical_experience_years": 15,
            "fellowship_training": 2,
            "peer_review_rating": 4.5,
            "annual_case_volume": 5000,
            "clinical_specialties": ["oncology", "hematology"]
        }),
        ("dr_jones", {
            "clinical_experience_years": 8,
            "fellowship_training": 1,
            "peer_review_rating": 3.8,
            "annual_case_volume": 2000,
            "clinical_specialties": ["pathology"]
        })
    ]
    
    scores = []
    for expert_id, credentials in experts:
        score = mkn.register_clinical_expert(expert_id, credentials)
        scores.append(score)
    
    print(f"  Registered experts: {len(mkn.expertise_network)}")
    print(f"  Expertise scores: {[f'{s:.2f}' for s in scores]}")
    
    return len(mkn.expertise_network) == 2 and scores[0] > scores[1]

def test_mkn_diagnostic_patterns():
    """Test MKN diagnostic pattern contribution."""
    print("Testing MKN diagnostic patterns...")
    
    mkn = MedicalKnowledgeNetwork()
    
    # Register expert
    mkn.register_clinical_expert("expert_oncologist", {
        "clinical_experience_years": 20,
        "fellowship_training": 3,
        "peer_review_rating": 4.8,
        "annual_case_volume": 8000
    })
    
    # Contribute diagnostic pattern
    pattern = {
        "disease_type": "breast_cancer",
        "symptoms": ["mass_detected", "irregular_borders", "microcalcifications"],
        "predicted_outcome": {
            "malignancy_probability": 0.85,
            "stage": "T2N0M0",
            "recommended_treatment": "surgical_resection"
        }
    }
    
    result = mkn.contribute_diagnostic_pattern("expert_oncologist", pattern)
    
    print(f"  Pattern contributed: {result.get('pattern_id') is not None}")
    print(f"  Expert weight applied: {result.get('expertise_weight', 0):.2f}")
    
    return "pattern_id" in result and result["expertise_weight"] > 1.0

def test_mkn_knowledge_query():
    """Test MKN knowledge querying."""
    print("Testing MKN knowledge query...")
    
    mkn = MedicalKnowledgeNetwork()
    
    # Register experts and add patterns
    mkn.register_clinical_expert("expert_1", {
        "clinical_experience_years": 15,
        "peer_review_rating": 4.2
    })
    
    mkn.register_clinical_expert("expert_2", {
        "clinical_experience_years": 25,
        "peer_review_rating": 4.7
    })
    
    # Add patterns
    pattern1 = {
        "disease_type": "lung_cancer",
        "symptoms": ["persistent_cough", "chest_pain", "weight_loss"],
        "predicted_outcome": {"malignancy_probability": 0.78}
    }
    
    pattern2 = {
        "disease_type": "lung_cancer", 
        "symptoms": ["persistent_cough", "hemoptysis", "dyspnea"],
        "predicted_outcome": {"malignancy_probability": 0.92}
    }
    
    mkn.contribute_diagnostic_pattern("expert_1", pattern1)
    mkn.contribute_diagnostic_pattern("expert_2", pattern2)
    
    # Query knowledge
    query = {
        "disease_type": "lung_cancer",
        "symptoms": ["persistent_cough", "chest_pain"]
    }
    
    result = mkn.query_diagnostic_knowledge(query)
    
    print(f"  Matched patterns: {len(result.get('matched_patterns', []))}")
    print(f"  Total patterns: {result.get('total_patterns', 0)}")
    
    return len(result.get("matched_patterns", [])) > 0

def test_mkn_consensus_diagnosis():
    """Test MKN consensus diagnosis generation."""
    print("Testing MKN consensus diagnosis...")
    
    mkn = MedicalKnowledgeNetwork()
    
    # Register multiple experts
    experts = ["expert_a", "expert_b", "expert_c"]
    for expert_id in experts:
        mkn.register_clinical_expert(expert_id, {
            "clinical_experience_years": 12,
            "peer_review_rating": 4.0
        })
    
    # Add diagnostic patterns
    patterns = [
        {"malignancy_probability": 0.80, "stage": 2},
        {"malignancy_probability": 0.85, "stage": 2}, 
        {"malignancy_probability": 0.75, "stage": 1}
    ]
    
    for i, expert_id in enumerate(experts):
        pattern = {
            "disease_type": "prostate_cancer",
            "predicted_outcome": patterns[i]
        }
        mkn.contribute_diagnostic_pattern(expert_id, pattern)
    
    # Generate consensus
    case_data = {"suspected_disease": "prostate_cancer"}
    consensus = mkn.generate_consensus_diagnosis(case_data)
    
    consensus_prob = consensus["consensus_diagnosis"].get("malignancy_probability", 0)
    
    print(f"  Contributing experts: {consensus.get('contributing_experts', 0)}")
    print(f"  Consensus probability: {consensus_prob:.3f}")
    print(f"  Confidence score: {consensus.get('confidence_score', 0):.3f}")
    
    return 0.75 < consensus_prob < 0.85  # Should be average of inputs

def run_mkn_tests():
    """Run all MKN tests."""
    print("🧠 Medical Knowledge Network (MKN) Testing")
    print("=" * 60)
    
    tests = [
        ("Expert Registration", test_mkn_expert_registration),
        ("Diagnostic Patterns", test_mkn_diagnostic_patterns),
        ("Knowledge Query", test_mkn_knowledge_query),
        ("Consensus Diagnosis", test_mkn_consensus_diagnosis),
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
    print(f"MKN Tests: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🏆 Medical Knowledge Network fully operational!")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_mkn_tests()
    exit(0 if success else 1)