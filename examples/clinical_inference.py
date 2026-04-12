"""
Clinical Inference Example

Demonstrates end-to-end clinical workflow for WSI analysis including:
- Multi-class disease classification
- Risk factor analysis
- Uncertainty quantification
- Clinical report generation
"""

import torch
import numpy as np
from pathlib import Path

from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.taxonomy import DiseaseTaxonomy
from src.clinical.risk_analysis import RiskAnalyzer
from src.clinical.uncertainty import UncertaintyQuantifier
from src.clinical.patient_context import ClinicalMetadata, PatientContextIntegrator
from src.clinical.reporting import ClinicalReportGenerator
from src.clinical.thresholds import ClinicalThresholdSystem


def main():
    """Run clinical inference example."""
    
    # 1. Load disease taxonomy
    print("Loading disease taxonomy...")
    taxonomy = DiseaseTaxonomy(config_file="configs/clinical/cancer_grading.yaml")
    print(f"Loaded taxonomy: {taxonomy.name} with {taxonomy.get_num_classes()} classes")
    
    # 2. Initialize models
    print("\nInitializing clinical models...")
    embedding_dim = 1024
    
    classifier = MultiClassDiseaseClassifier(
        embedding_dim=embedding_dim,
        taxonomy=taxonomy,
        hidden_dim=512,
        dropout=0.3
    )
    
    risk_analyzer = RiskAnalyzer(
        embedding_dim=embedding_dim,
        num_disease_states=taxonomy.get_num_classes(),
        time_horizons=["1_year", "5_year", "10_year"]
    )
    
    uncertainty_quantifier = UncertaintyQuantifier(
        num_classes=taxonomy.get_num_classes(),
        calibration_method="temperature"
    )
    
    # 3. Load threshold system
    print("\nLoading clinical thresholds...")
    threshold_system = ClinicalThresholdSystem()
    threshold_system.load_from_file("configs/clinical/cancer_grading.yaml")
    
    # 4. Create patient context
    print("\nCreating patient context...")
    clinical_metadata = ClinicalMetadata(
        age=65,
        sex="M",
        smoking_status="former",
        alcohol_consumption="moderate",
        medications=["aspirin", "metformin"],
        exercise_frequency="light",
        family_history={"cancer": True, "heart_disease": False}
    )
    
    # 5. Simulate WSI features (in practice, extract from actual WSI)
    print("\nSimulating WSI features...")
    batch_size = 1
    num_patches = 100
    wsi_features = torch.randn(batch_size, num_patches, embedding_dim)
    
    # Average pool for slide-level representation
    slide_embedding = wsi_features.mean(dim=1)  # [batch, embedding_dim]
    
    # 6. Run inference
    print("\nRunning clinical inference...")
    
    # Classification
    with torch.no_grad():
        logits = classifier(slide_embedding)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get primary diagnosis
        primary_diagnosis_idx = torch.argmax(probabilities, dim=1).item()
        primary_diagnosis_prob = probabilities[0, primary_diagnosis_idx].item()
        
        disease_ids = list(taxonomy.diseases.keys())
        primary_diagnosis = disease_ids[primary_diagnosis_idx]
        
        print(f"\nPrimary Diagnosis: {primary_diagnosis}")
        print(f"Confidence: {primary_diagnosis_prob:.2%}")
        
        # Get top-3 diagnoses
        top3_probs, top3_indices = torch.topk(probabilities[0], k=3)
        print("\nTop 3 Diagnoses:")
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices), 1):
            disease_id = disease_ids[idx.item()]
            print(f"  {i}. {disease_id}: {prob.item():.2%}")
        
        # Risk analysis
        risk_scores = risk_analyzer(slide_embedding)
        print(f"\nRisk Scores (shape: {risk_scores.shape}):")
        print(f"  1-year risk: {risk_scores[0, primary_diagnosis_idx, 0].item():.2%}")
        print(f"  5-year risk: {risk_scores[0, primary_diagnosis_idx, 1].item():.2%}")
        print(f"  10-year risk: {risk_scores[0, primary_diagnosis_idx, 2].item():.2%}")
        
        # Uncertainty quantification
        uncertainty_result = uncertainty_quantifier(logits)
        calibrated_probs = uncertainty_result['calibrated_probabilities']
        uncertainty_scores = uncertainty_result['uncertainty_scores']
        
        print(f"\nUncertainty Analysis:")
        print(f"  Primary diagnosis uncertainty: {uncertainty_scores[0, primary_diagnosis_idx].item():.3f}")
        print(f"  Calibrated probability: {calibrated_probs[0, primary_diagnosis_idx].item():.2%}")
        
        # Threshold evaluation
        flagged = threshold_system.evaluate_risk_scores(risk_scores)
        if flagged[0]:
            print("\n⚠️  Case flagged for physician review (exceeds risk thresholds)")
        else:
            print("\n✓ Case within normal risk thresholds")
    
    # 7. Generate clinical report
    print("\nGenerating clinical report...")
    report_generator = ClinicalReportGenerator(
        template_dir="configs/clinical/templates"
    )
    
    report_data = {
        "patient_id": "ANON_12345",
        "scan_date": "2026-04-12",
        "primary_diagnosis": primary_diagnosis,
        "primary_probability": primary_diagnosis_prob,
        "top_diagnoses": [
            {"disease": disease_ids[idx.item()], "probability": prob.item()}
            for prob, idx in zip(top3_probs, top3_indices)
        ],
        "risk_scores": {
            "1_year": risk_scores[0, primary_diagnosis_idx, 0].item(),
            "5_year": risk_scores[0, primary_diagnosis_idx, 1].item(),
            "10_year": risk_scores[0, primary_diagnosis_idx, 2].item()
        },
        "uncertainty": uncertainty_scores[0, primary_diagnosis_idx].item(),
        "flagged_for_review": flagged[0].item(),
        "clinical_metadata": clinical_metadata.to_dict()
    }
    
    print("\nClinical Report Summary:")
    print(f"  Patient: {report_data['patient_id']}")
    print(f"  Scan Date: {report_data['scan_date']}")
    print(f"  Primary Diagnosis: {report_data['primary_diagnosis']} ({report_data['primary_probability']:.2%})")
    print(f"  Review Required: {'Yes' if report_data['flagged_for_review'] else 'No'}")
    
    print("\n✓ Clinical inference complete!")


if __name__ == "__main__":
    main()
