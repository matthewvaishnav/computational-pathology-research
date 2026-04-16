#!/usr/bin/env python3
"""
Example script demonstrating treatment response monitoring functionality.

This script shows how to use the TreatmentResponseAnalyzer to:
1. Compute comprehensive treatment response metrics
2. Analyze treatment response trajectories
3. Identify unexpected responses
4. Compare therapeutic regimens
5. Generate clinical reports

Usage:
    python examples/treatment_response_example.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical.treatment_response import TreatmentResponseAnalyzer, TreatmentResponseMetrics
from clinical.longitudinal import (
    LongitudinalTracker,
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    TreatmentResponseCategory,
)
from clinical.taxonomy import DiseaseTaxonomy


def create_sample_data():
    """Create sample patient data for demonstration."""
    print("Creating sample patient data...")

    # Create sample taxonomy
    taxonomy_config = {
        "name": "cancer_grading",
        "version": "1.0",
        "disease_states": {
            "benign": {"level": 0, "parent": None},
            "low_grade_cancer": {"level": 1, "parent": "benign"},
            "intermediate_grade_cancer": {"level": 2, "parent": "low_grade_cancer"},
            "high_grade_cancer": {"level": 3, "parent": "intermediate_grade_cancer"},
            "metastatic_cancer": {"level": 4, "parent": "high_grade_cancer"},
        },
    }
    taxonomy = DiseaseTaxonomy.from_dict(taxonomy_config)

    # Create longitudinal tracker
    longitudinal_tracker = LongitudinalTracker(taxonomy=taxonomy)

    # Create sample patient timelines
    timelines = []

    for patient_idx in range(3):
        patient_id = f"patient_{patient_idx:03d}"
        timeline = PatientTimeline(patient_id)

        # Add baseline scan
        baseline_date = datetime(2024, 1, 1) + timedelta(days=patient_idx * 10)
        baseline_scan = ScanRecord(
            scan_id=f"scan_{patient_idx}_baseline",
            scan_date=baseline_date,
            disease_state="high_grade_cancer",
            disease_probabilities={
                "benign": 0.05,
                "low_grade_cancer": 0.10,
                "intermediate_grade_cancer": 0.15,
                "high_grade_cancer": 0.70,
                "metastatic_cancer": 0.00,
            },
            confidence=0.85 + patient_idx * 0.05,
        )
        timeline.add_scan(baseline_scan)

        # Add treatment
        treatment_date = baseline_date + timedelta(days=7)
        treatment_types = ["chemotherapy", "immunotherapy", "radiation"]
        treatment = TreatmentEvent(
            treatment_id=f"treatment_{patient_idx}_001",
            treatment_date=treatment_date,
            treatment_type=treatment_types[patient_idx],
            treatment_details={"regimen": f"Standard {treatment_types[patient_idx]} protocol"},
        )
        timeline.add_treatment(treatment)

        # Add response scan with different outcomes
        response_date = treatment_date + timedelta(days=30 + patient_idx * 15)

        if patient_idx == 0:  # Good response
            response_probs = {
                "benign": 0.20,
                "low_grade_cancer": 0.50,
                "intermediate_grade_cancer": 0.25,
                "high_grade_cancer": 0.05,
                "metastatic_cancer": 0.00,
            }
            response_state = "low_grade_cancer"
        elif patient_idx == 1:  # Partial response
            response_probs = {
                "benign": 0.10,
                "low_grade_cancer": 0.20,
                "intermediate_grade_cancer": 0.45,
                "high_grade_cancer": 0.25,
                "metastatic_cancer": 0.00,
            }
            response_state = "intermediate_grade_cancer"
        else:  # Progressive disease
            response_probs = {
                "benign": 0.02,
                "low_grade_cancer": 0.05,
                "intermediate_grade_cancer": 0.08,
                "high_grade_cancer": 0.35,
                "metastatic_cancer": 0.50,
            }
            response_state = "metastatic_cancer"

        response_scan = ScanRecord(
            scan_id=f"scan_{patient_idx}_response",
            scan_date=response_date,
            disease_state=response_state,
            disease_probabilities=response_probs,
            confidence=0.80 + patient_idx * 0.05,
        )
        timeline.add_scan(response_scan)

        # Add follow-up scan
        followup_date = response_date + timedelta(days=60)
        followup_scan = ScanRecord(
            scan_id=f"scan_{patient_idx}_followup",
            scan_date=followup_date,
            disease_state=response_state,  # Assume stable for simplicity
            disease_probabilities=response_probs,
            confidence=0.82 + patient_idx * 0.03,
        )
        timeline.add_scan(followup_scan)

        longitudinal_tracker.register_timeline(timeline)
        timelines.append(timeline)

    return taxonomy, longitudinal_tracker, timelines


def demonstrate_treatment_response_analysis():
    """Demonstrate comprehensive treatment response analysis."""
    print("\n" + "=" * 60)
    print("TREATMENT RESPONSE MONITORING DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    taxonomy, longitudinal_tracker, timelines = create_sample_data()

    # Initialize treatment response analyzer
    analyzer = TreatmentResponseAnalyzer(
        longitudinal_tracker=longitudinal_tracker,
        taxonomy=taxonomy,
    )

    print(f"\nAnalyzing treatment responses for {len(timelines)} patients...")

    # Analyze each patient's treatment response
    all_metrics = []

    for i, timeline in enumerate(timelines):
        print(f"\n--- Patient {i+1} Analysis ---")

        # Get treatment ID
        treatments = timeline.get_treatments()
        if not treatments:
            continue

        treatment_id = treatments[0].treatment_id
        treatment_type = treatments[0].treatment_type

        # Compute comprehensive metrics
        try:
            metrics = analyzer.compute_treatment_response_metrics(
                timeline=timeline,
                treatment_id=treatment_id,
            )
            all_metrics.append(metrics)

            print(f"Treatment Type: {treatment_type}")
            print(f"Response Category: {metrics.response_category.value}")
            print(f"Response Kinetics: {metrics.response_kinetics.value}")
            print(f"Days to Response: {metrics.days_to_response}")
            print(f"Probability Change: {metrics.probability_change:.3f}")
            print(f"Response Magnitude: {metrics.response_magnitude:.3f}")
            print(f"Response Consistency: {metrics.response_consistency:.3f}")
            print(f"Unexpected Response: {metrics.is_unexpected}")

            if metrics.is_unexpected:
                print(f"Unexpected Type: {metrics.unexpected_type.value}")
                print(f"Unexpected Score: {metrics.unexpected_score:.3f}")

            # Analyze trajectory
            trajectory_data = analyzer.analyze_treatment_response_trajectory(
                timeline=timeline,
                treatment_id=treatment_id,
            )

            print(f"Trajectory Pattern: {trajectory_data['trajectory_pattern']}")
            print(f"Number of Response Phases: {len(trajectory_data['response_phases'])}")

        except Exception as e:
            print(f"Error analyzing patient {i+1}: {e}")
            continue

    # Demonstrate unexpected response detection
    print(f"\n--- Unexpected Response Detection ---")
    unexpected_cases = analyzer.identify_unexpected_responses(timelines)
    print(f"Found {len(unexpected_cases)} unexpected response cases")

    for case in unexpected_cases:
        print(f"Patient: {case['patient_id_hash'][:8]}...")
        print(f"Treatment: {case['treatment_type']}")
        print(f"Unexpected Type: {case['unexpected_type']}")
        print(f"Unexpected Score: {case['unexpected_score']:.3f}")
        print()

    # Demonstrate patient factor correlation
    if len(all_metrics) >= 3:
        print(f"--- Patient Factor Correlation Analysis ---")
        correlation_results = analyzer.correlate_response_with_patient_factors(all_metrics)

        print(f"Sample Size: {correlation_results['sample_size']}")
        print(f"Factors Analyzed: {correlation_results['factors_analyzed']}")

        response_dist = correlation_results["response_distribution"]
        print("Response Distribution:")
        for category, count in response_dist.items():
            print(f"  {category}: {count}")

    # Demonstrate regimen comparison
    if len(all_metrics) >= 2:
        print(f"\n--- Therapeutic Regimen Comparison ---")
        comparison_results = analyzer.compare_therapeutic_regimens(all_metrics)

        print(f"Total Regimens: {comparison_results['total_regimens']}")
        print(f"Total Cases: {comparison_results['total_cases']}")

        print("\nRegimen Ranking (by effectiveness):")
        for rank, regimen_info in enumerate(comparison_results["regimen_ranking"], 1):
            print(
                f"{rank}. {regimen_info['regimen']}: "
                f"Score={regimen_info['effectiveness_score']:.3f}, "
                f"Response Rate={regimen_info['overall_response_rate']:.3f}"
            )

    # Generate sample clinical report
    if all_metrics:
        print(f"\n--- Sample Clinical Report ---")
        sample_metrics = all_metrics[0]

        # Get trajectory data for the first patient
        first_timeline = timelines[0]
        first_treatment_id = first_timeline.get_treatments()[0].treatment_id
        trajectory_data = analyzer.analyze_treatment_response_trajectory(
            timeline=first_timeline,
            treatment_id=first_treatment_id,
        )

        report = analyzer.generate_treatment_response_report(
            metrics=sample_metrics,
            trajectory_data=trajectory_data,
        )

        print(f"Patient ID: {report['patient_id_hash'][:8]}...")
        print(f"Treatment ID: {report['treatment_id']}")

        summary = report["executive_summary"]
        print(f"\nExecutive Summary:")
        print(f"  Response Category: {summary['response_category']}")
        print(f"  Response Kinetics: {summary['response_kinetics']}")
        print(f"  Clinical Significance: {summary['clinical_significance']}")

        interpretation = report["clinical_interpretation"]
        print(f"\nClinical Recommendations:")
        for i, rec in enumerate(interpretation["recommendations"], 1):
            print(f"  {i}. {rec}")

        if interpretation["uncertainty_factors"]:
            print(f"\nUncertainty Factors:")
            for factor in interpretation["uncertainty_factors"]:
                print(f"  - {factor}")

    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_treatment_response_analysis()
