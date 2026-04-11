"""
Demo: Multi-class disease state classifier with MIL models.

This example demonstrates how to use the MultiClassDiseaseClassifier
with existing MIL models (AttentionMIL, CLAM, TransMIL) and multimodal
fusion for clinical disease classification.
"""

import torch
import torch.nn as nn

from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.taxonomy import DiseaseTaxonomy
from src.models.multimodal import MultimodalFusionModel


def create_cancer_grading_taxonomy():
    """Create a cancer grading taxonomy."""
    return DiseaseTaxonomy(
        config_dict={
            "name": "Cancer Grading System",
            "version": "1.0",
            "description": "Multi-class cancer grading classification",
            "diseases": [
                {
                    "id": "benign",
                    "name": "Benign",
                    "description": "Non-cancerous tissue",
                    "parent": None,
                    "children": [],
                },
                {
                    "id": "grade_1",
                    "name": "Grade 1 Cancer",
                    "description": "Well-differentiated cancer",
                    "parent": None,
                    "children": [],
                },
                {
                    "id": "grade_2",
                    "name": "Grade 2 Cancer",
                    "description": "Moderately differentiated cancer",
                    "parent": None,
                    "children": [],
                },
                {
                    "id": "grade_3",
                    "name": "Grade 3 Cancer",
                    "description": "Poorly differentiated cancer",
                    "parent": None,
                    "children": [],
                },
            ],
        }
    )


def demo_basic_classifier():
    """Demo 1: Basic classifier usage."""
    print("=" * 80)
    print("Demo 1: Basic Multi-Class Disease Classifier")
    print("=" * 80)

    # Create taxonomy
    taxonomy = create_cancer_grading_taxonomy()
    print(f"\nTaxonomy: {taxonomy.name}")
    print(f"Number of classes: {taxonomy.get_num_classes()}")
    print(f"Disease states: {taxonomy.disease_ids}")

    # Create classifier
    classifier = MultiClassDiseaseClassifier(
        taxonomy=taxonomy, input_dim=256, hidden_dim=128, dropout=0.3
    )
    print(f"\n{classifier}")

    # Simulate input embeddings (batch of 4 samples)
    batch_size = 4
    embeddings = torch.randn(batch_size, 256)

    # Forward pass
    output = classifier(embeddings)

    # Display results
    print("\nPrediction Results:")
    print("-" * 80)
    for i in range(batch_size):
        print(f"\nSample {i + 1}:")
        print("  Probabilities:")
        for disease_id in taxonomy.disease_ids:
            idx = classifier.id_to_idx[disease_id]
            prob = output["probabilities"][i, idx].item()
            print(f"    {disease_id:12s}: {prob:.4f}")

        primary_idx = output["primary_diagnosis"][i].item()
        primary_id = classifier.idx_to_id[primary_idx]
        confidence = output["confidence"][i].item()
        print(f"  Primary Diagnosis: {primary_id} (confidence: {confidence:.4f})")

    # Verify probability distribution properties
    prob_sums = output["probabilities"].sum(dim=1)
    print(f"\nProbability sum verification: {prob_sums}")
    print(f"All sums equal 1.0: {torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)}")


def demo_with_attention_mil():
    """Demo 2: Integration with AttentionMIL."""
    print("\n" + "=" * 80)
    print("Demo 2: Integration with AttentionMIL")
    print("=" * 80)

    # Create taxonomy
    taxonomy = create_cancer_grading_taxonomy()

    # Create AttentionMIL model (for feature aggregation)
    # Note: AttentionMIL outputs aggregated features, not final predictions
    feature_dim = 1024
    hidden_dim = 256

    # Simple encoder to simulate AttentionMIL aggregation
    mil_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))

    # Create disease classifier
    classifier = MultiClassDiseaseClassifier(
        taxonomy=taxonomy, input_dim=hidden_dim, hidden_dim=128
    )

    print(f"\nMIL Encoder output dim: {hidden_dim}")
    print(f"Classifier input dim: {classifier.input_dim}")
    print(f"Number of disease classes: {classifier.num_classes}")

    # Simulate WSI patch features (batch of 2 slides, 100 patches each)
    batch_size = 2
    num_patches = 100
    patch_features = torch.randn(batch_size, num_patches, feature_dim)

    # Aggregate patches (simplified - real AttentionMIL uses attention)
    aggregated_features = mil_encoder(patch_features.mean(dim=1))

    # Classify disease state
    _ = classifier(aggregated_features)

    # Display results
    print("\nClassification Results:")
    print("-" * 80)
    disease_ids, probs, confidence = classifier.predict_disease_ids(aggregated_features)

    for i in range(batch_size):
        print(f"\nSlide {i + 1}:")
        print(f"  Primary Diagnosis: {disease_ids[i]}")
        print(f"  Confidence: {confidence[i].item():.4f}")
        print("  Full probability distribution:")
        for disease_id in taxonomy.disease_ids:
            idx = classifier.id_to_idx[disease_id]
            prob = probs[i, idx].item()
            bar = "█" * int(prob * 50)
            print(f"    {disease_id:12s}: {prob:.4f} {bar}")


def demo_with_multimodal_fusion():
    """Demo 3: Integration with multimodal fusion."""
    print("\n" + "=" * 80)
    print("Demo 3: Integration with Multimodal Fusion")
    print("=" * 80)

    # Create taxonomy
    taxonomy = create_cancer_grading_taxonomy()

    # Create multimodal fusion model
    fusion_model = MultimodalFusionModel(embed_dim=256, dropout=0.1)

    # Create disease classifier
    classifier = MultiClassDiseaseClassifier(taxonomy=taxonomy, input_dim=256, hidden_dim=128)

    print(f"\nFusion model output dim: {fusion_model.get_embedding_dim()}")
    print(f"Classifier input dim: {classifier.input_dim}")

    # Simulate multimodal input
    batch_size = 3
    batch = {
        "wsi_features": torch.randn(batch_size, 50, 1024),  # WSI patches
        "genomic": torch.randn(batch_size, 2000),  # Genomic features
        "clinical_text": torch.randint(0, 30000, (batch_size, 128)),  # Clinical notes
    }

    # Fuse modalities
    fused_embedding = fusion_model(batch)
    print(f"\nFused embedding shape: {fused_embedding.shape}")

    # Classify disease state
    _ = classifier(fused_embedding)

    # Get top-3 differential diagnoses
    top_k_ids, top_k_probs = classifier.get_top_k_diagnoses(fused_embedding, k=3)

    # Display results
    print("\nMultimodal Classification Results:")
    print("-" * 80)
    for i in range(batch_size):
        print(f"\nPatient {i + 1}:")
        print("  Differential Diagnosis (Top 3):")
        for rank, (disease_id, prob) in enumerate(zip(top_k_ids[i], top_k_probs[i]), 1):
            print(f"    {rank}. {disease_id:12s}: {prob.item():.4f}")


def demo_taxonomy_update():
    """Demo 4: Dynamic taxonomy update."""
    print("\n" + "=" * 80)
    print("Demo 4: Dynamic Taxonomy Update")
    print("=" * 80)

    # Create initial simple taxonomy
    simple_taxonomy = DiseaseTaxonomy(
        config_dict={
            "name": "Simple Binary Classification",
            "version": "1.0",
            "diseases": [
                {"id": "normal", "name": "Normal", "parent": None, "children": []},
                {"id": "abnormal", "name": "Abnormal", "parent": None, "children": []},
            ],
        }
    )

    # Create classifier
    classifier = MultiClassDiseaseClassifier(taxonomy=simple_taxonomy, input_dim=256)

    print(f"\nInitial taxonomy: {simple_taxonomy.name}")
    print(f"Number of classes: {classifier.num_classes}")
    print(f"Disease states: {classifier.disease_ids}")

    # Test with initial taxonomy
    embeddings = torch.randn(2, 256)
    output1 = classifier(embeddings)
    print(f"\nOutput shape with initial taxonomy: {output1['probabilities'].shape}")

    # Update to complex taxonomy
    complex_taxonomy = create_cancer_grading_taxonomy()
    classifier.update_taxonomy(complex_taxonomy)

    print(f"\nUpdated taxonomy: {complex_taxonomy.name}")
    print(f"Number of classes: {classifier.num_classes}")
    print(f"Disease states: {classifier.disease_ids}")

    # Test with updated taxonomy
    output2 = classifier(embeddings)
    print(f"\nOutput shape with updated taxonomy: {output2['probabilities'].shape}")

    # Display new predictions
    print("\nPredictions with updated taxonomy:")
    print("-" * 80)
    for i in range(2):
        primary_idx = output2["primary_diagnosis"][i].item()
        primary_id = classifier.idx_to_id[primary_idx]
        confidence = output2["confidence"][i].item()
        print(f"Sample {i + 1}: {primary_id} (confidence: {confidence:.4f})")


def demo_disease_probability_queries():
    """Demo 5: Querying disease probabilities."""
    print("\n" + "=" * 80)
    print("Demo 5: Disease Probability Queries")
    print("=" * 80)

    # Create taxonomy
    taxonomy = create_cancer_grading_taxonomy()

    # Create classifier
    classifier = MultiClassDiseaseClassifier(taxonomy=taxonomy, input_dim=256)

    # Simulate embeddings
    embeddings = torch.randn(1, 256)

    # Get disease probabilities by ID
    disease_probs = classifier.get_disease_probabilities(embeddings)

    print("\nDisease Probability Query Results:")
    print("-" * 80)
    print("\nProbability for each disease state:")
    for disease_id, prob in disease_probs.items():
        disease_info = taxonomy.get_disease(disease_id)
        print(f"  {disease_id:12s}: {prob.item():.4f} - {disease_info['description']}")

    # Get taxonomy information
    info = classifier.get_taxonomy_info()
    print("\nTaxonomy Information:")
    print(f"  Name: {info['name']}")
    print(f"  Version: {info['version']}")
    print(f"  Total classes: {info['num_classes']}")
    print(f"  Root diseases: {info['root_diseases']}")
    print(f"  Leaf diseases: {info['leaf_diseases']}")


if __name__ == "__main__":
    # Run all demos
    demo_basic_classifier()
    demo_with_attention_mil()
    demo_with_multimodal_fusion()
    demo_taxonomy_update()
    demo_disease_probability_queries()

    print("\n" + "=" * 80)
    print("All demos completed successfully!")
    print("=" * 80)
