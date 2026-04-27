"""
Basic test for Phase 2 explainability components
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.explainability.uncertainty_quantification import (
        UncertaintyQuantificationSystem, MonteCarloDropout
    )
    print("✓ Uncertainty quantification module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import uncertainty quantification: {e}")

try:
    from src.foundation.multi_disease_model import create_foundation_model
    print("✓ Foundation model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import foundation model: {e}")

# Test basic functionality without external dependencies
def test_basic_uncertainty():
    """Test basic uncertainty quantification without FAISS"""
    try:
        # Create a simple model
        model = create_foundation_model()
        print("✓ Foundation model created successfully")
        
        # Test Monte Carlo dropout
        mc_dropout = MonteCarloDropout(model, num_samples=3)
        patches = torch.randn(1, 10, 3, 224, 224)
        
        uncertainty_metrics = mc_dropout.estimate_uncertainty(
            patches, disease_type="breast"
        )
        
        print(f"✓ Uncertainty estimation completed:")
        print(f"  - Total uncertainty: {uncertainty_metrics.total_uncertainty:.4f}")
        print(f"  - Epistemic uncertainty: {uncertainty_metrics.epistemic_uncertainty:.4f}")
        print(f"  - Aleatoric uncertainty: {uncertainty_metrics.aleatoric_uncertainty:.4f}")
        print(f"  - Confidence interval: {uncertainty_metrics.confidence_interval}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic uncertainty test failed: {e}")
        return False

def test_counterfactual_basic():
    """Test basic counterfactual explanation without external dependencies"""
    try:
        from src.explainability.counterfactual_explanations import (
            BiologicalPlausibilityValidator
        )
        
        validator = BiologicalPlausibilityValidator("breast")
        
        original_features = torch.randn(2048)
        modified_features = original_features + 0.1 * torch.randn(2048)
        
        validation_results = validator.validate_changes(
            original_features, modified_features
        )
        
        print(f"✓ Biological plausibility validation completed:")
        print(f"  - Overall plausibility: {validation_results['overall_plausibility']:.4f}")
        print(f"  - Morphology score: {validation_results['morphology_score']:.4f}")
        print(f"  - Magnitude score: {validation_results['magnitude_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Counterfactual test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Phase 2 Explainability Components")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    if test_basic_uncertainty():
        success_count += 1
    
    if test_counterfactual_basic():
        success_count += 1
    
    print("=" * 50)
    print(f"Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✓ All basic tests passed! Phase 2 core components are working.")
    else:
        print("✗ Some tests failed. Check the error messages above.")