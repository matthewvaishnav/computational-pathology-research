"""
Example demonstrating risk factor analysis for disease development prediction.

This example shows how to:
1. Create a disease taxonomy
2. Initialize the RiskAnalyzer
3. Calculate risk scores with imaging features and clinical metadata
4. Use clinical decision thresholds to flag high-risk cases
"""

import torch

from src.clinical import (
    ClinicalMetadata,
    ClinicalThresholdSystem,
    DiseaseTaxonomy,
    RiskAnalyzer,
    Sex,
    SmokingStatus,
    ThresholdConfig,
)


def main():
    """Run risk analysis example."""
    print("=" * 80)
    print("Risk Factor Analysis Example")
    print("=" * 80)

    # 1. Create disease taxonomy
    print("\n1. Creating disease taxonomy...")
    taxonomy = DiseaseTaxonomy(
        config_dict={
            "name": "Cancer Risk Assessment",
            "version": "1.0",
            "diseases": [
                {"id": "benign", "name": "Benign", "parent": None, "children": []},
                {"id": "grade_1", "name": "Grade 1 Cancer", "parent": None, "children": []},
                {"id": "grade_2", "name": "Grade 2 Cancer", "parent": None, "children": []},
                {"id": "grade_3", "name": "Grade 3 Cancer", "parent": None, "children": []},
            ],
        }
    )
    print(f"   Created taxonomy: {taxonomy.name}")
    print(f"   Number of disease states: {taxonomy.get_num_classes()}")

    # 2. Initialize RiskAnalyzer
    print("\n2. Initializing RiskAnalyzer...")
    analyzer = RiskAnalyzer(
        taxonomy=taxonomy,
        input_dim=256,
        hidden_dim=128,
        time_horizons=[1, 5, 10],  # 1-year, 5-year, 10-year risk
    )
    print(f"   {analyzer}")
    print(f"   Time horizons: {analyzer.get_time_horizon_names()}")

    # 3. Create sample clinical metadata
    print("\n3. Creating sample patient clinical metadata...")
    clinical_metadata = [
        ClinicalMetadata(
            age=65,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.FORMER,
            family_history=["cancer", "diabetes"],
        ),
        ClinicalMetadata(
            age=45,
            sex=Sex.FEMALE,
            smoking_status=SmokingStatus.NEVER,
            family_history=[],
        ),
        ClinicalMetadata(
            age=70,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.CURRENT,
            family_history=["cancer", "heart_disease", "hypertension"],
        ),
    ]

    for i, metadata in enumerate(clinical_metadata):
        print(f"   Patient {i+1}: {metadata}")

    # 4. Generate sample imaging features (in practice, these come from WSI encoder)
    print("\n4. Generating sample imaging features...")
    batch_size = len(clinical_metadata)
    imaging_features = torch.randn(batch_size, 256)
    print(f"   Imaging features shape: {imaging_features.shape}")

    # 5. Calculate risk scores
    print("\n5. Calculating risk scores...")
    analyzer.eval()
    with torch.no_grad():
        risk_output = analyzer(imaging_features, clinical_metadata=clinical_metadata)

    print(f"   Risk scores shape: {risk_output['risk_scores'].shape}")
    print(f"   Anomaly scores shape: {risk_output['anomaly_scores'].shape}")

    # Display risk scores for each patient
    print("\n   Risk Scores by Patient:")
    risk_scores = risk_output["risk_scores"]
    anomaly_scores = risk_output["anomaly_scores"]

    for i in range(batch_size):
        print(f"\n   Patient {i+1}:")
        print(f"      Age: {clinical_metadata[i].age}, Smoking: {clinical_metadata[i].smoking_status.value}")

        for j, disease_id in enumerate(taxonomy.disease_ids):
            print(f"      {disease_id}:")
            for k, horizon in enumerate(analyzer.time_horizons):
                risk_score = risk_scores[i, j, k].item()
                print(f"         {horizon}-year risk: {risk_score:.3f}")

        print(f"      Anomaly scores:")
        for j, disease_id in enumerate(taxonomy.disease_ids):
            anomaly_score = anomaly_scores[i, j].item()
            print(f"         {disease_id}: {anomaly_score:.3f}")

    # 6. Get risk by disease ID
    print("\n6. Getting risk scores organized by disease ID...")
    risk_by_disease = analyzer.get_risk_by_disease_id(
        imaging_features, clinical_metadata=clinical_metadata
    )

    print(f"   Available disease IDs: {list(risk_by_disease.keys())}")
    print(f"   Example - 'grade_3' risk scores shape: {risk_by_disease['grade_3'].shape}")

    # 7. Set up clinical decision thresholds
    print("\n7. Setting up clinical decision thresholds...")
    threshold_system = ClinicalThresholdSystem(
        default_risk_threshold=0.5,
        default_confidence_threshold=0.7,
        default_anomaly_threshold=0.6,
    )

    # Add disease-specific thresholds (lower threshold for higher grade cancers)
    threshold_system.add_threshold(
        ThresholdConfig(
            disease_id="grade_3",
            risk_threshold=0.3,  # Lower threshold for aggressive cancer
            anomaly_threshold=0.5,
        )
    )

    threshold_system.add_threshold(
        ThresholdConfig(
            disease_id="grade_2",
            risk_threshold=0.4,
            anomaly_threshold=0.55,
        )
    )

    print(f"   {threshold_system}")

    # 8. Identify high-risk cases
    print("\n8. Identifying high-risk cases...")
    high_risk_mask, flagged_details = analyzer.get_high_risk_cases(
        imaging_features, clinical_metadata=clinical_metadata, threshold=0.5
    )

    print(f"   High-risk cases: {high_risk_mask.sum().item()} out of {batch_size}")

    for i in range(batch_size):
        if high_risk_mask[i]:
            print(f"   ⚠️  Patient {i+1} flagged as high-risk")

    # 9. Evaluate with threshold system
    print("\n9. Evaluating cases with threshold system...")
    risk_flags = threshold_system.evaluate_risk_scores(
        risk_scores, disease_ids=taxonomy.disease_ids
    )

    anomaly_flags = threshold_system.evaluate_anomaly_scores(
        anomaly_scores, disease_ids=taxonomy.disease_ids
    )

    print(f"   Risk flags: {risk_flags}")
    print(f"   Anomaly flags: {anomaly_flags}")

    for i in range(batch_size):
        flags = []
        if risk_flags[i]:
            flags.append("HIGH_RISK")
        if anomaly_flags[i]:
            flags.append("ANOMALOUS")

        if flags:
            print(f"   Patient {i+1}: {', '.join(flags)} - Requires physician review")
        else:
            print(f"   Patient {i+1}: Normal - No flags")

    print("\n" + "=" * 80)
    print("Risk analysis example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
