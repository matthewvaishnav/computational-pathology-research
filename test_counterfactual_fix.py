"""
Test counterfactual fix
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.foundation.multi_disease_model import create_foundation_model
from src.explainability.counterfactual_explanations import CounterfactualExplanationSystem

def test_counterfactual():
    """Test counterfactual generation"""
    
    # Create model and system
    model = create_foundation_model()
    counterfactual_system = CounterfactualExplanationSystem(
        generator_type="gradient",
        max_iterations=5,  # Reduced for testing
        learning_rate=0.1
    )
    
    # Generate test data
    patches = torch.randn(1, 10, 3, 224, 224)
    
    with torch.no_grad():
        prediction = model(patches, disease_type="breast")
    
    print("Testing counterfactual generation...")
    
    try:
        counterfactual = counterfactual_system.generate_explanation(
            model, patches, prediction, disease_type="breast"
        )
        
        if counterfactual:
            print("✓ Counterfactual generated successfully!")
            print(f"  Original: {counterfactual.original_prediction}")
            print(f"  Target: {counterfactual.target_prediction}")
            print(f"  Plausibility: {counterfactual.plausibility_score:.4f}")
            print(f"  Required changes: {counterfactual.required_changes}")
        else:
            print("✗ Counterfactual generation returned None")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_counterfactual()