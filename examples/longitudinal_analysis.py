"""
Longitudinal Patient Tracking Example

Demonstrates longitudinal patient tracking including:
- Patient timeline management
- Disease progression tracking
- Treatment response monitoring
- Timeline visualization
"""

import torch
import numpy as np
from datetime import datetime, timedelta

from src.clinical.longitudinal import PatientTimeline, LongitudinalTracker, ScanRecord, TreatmentEvent
from src.clinical.treatment_response import TreatmentResponseAnalyzer, TreatmentResponseCategory
from src.visualization.timeline import TimelineVisualizer


def main():
    """Run longitudinal tracking example."""
    
    print("Longitudinal Patient Tracking Example")
    print("=" * 50)
    
    # 1. Create patient timeline
    print("\n1. Creating patient timeline...")
    patient_id = "PATIENT_001"
    timeline = PatientTimeline(patient_id=patient_id)
    
    # 2. Add historical scans
    print("\n2. Adding historical scans...")
    base_date = datetime(2024, 1, 15)
    
    # Scan 1: Initial diagnosis
    scan1 = ScanRecord(
        scan_id="SCAN_001",
        scan_date=(base_date).isoformat(),
        disease_state="grade_2",
        disease_probabilities={"normal": 0.05, "benign": 0.10, "grade_1": 0.15, "grade_2": 0.60, "grade_3": 0.10},
        risk_scores={"1_year": 0.55, "5_year": 0.65, "10_year": 0.70},
        model_version="v1.0.0"
    )
    timeline.add_scan(scan1)
    print(f"  Added scan 1: {scan1.disease_state} (confidence: {scan1.disease_probabilities[scan1.disease_state]:.2%})")
    
    # Treatment event
    treatment1 = TreatmentEvent(
        treatment_id="TX_001",
        treatment_date=(base_date + timedelta(days=7)).isoformat(),
        treatment_type="chemotherapy",
        treatment_details={"regimen": "FOLFOX", "cycles": 6}
    )
    timeline.add_treatment(treatment1)
    print(f"  Added treatment: {treatment1.treatment_type}")
    
    # Scan 2: Post-treatment (3 months)
    scan2 = ScanRecord(
        scan_id="SCAN_002",
        scan_date=(base_date + timedelta(days=90)).isoformat(),
        disease_state="grade_1",
        disease_probabilities={"normal": 0.10, "benign": 0.20, "grade_1": 0.55, "grade_2": 0.10, "grade_3": 0.05},
        risk_scores={"1_year": 0.35, "5_year": 0.45, "10_year": 0.50},
        model_version="v1.0.0"
    )
    timeline.add_scan(scan2)
    print(f"  Added scan 2: {scan2.disease_state} (confidence: {scan2.disease_probabilities[scan2.disease_state]:.2%})")
    
    # Scan 3: Follow-up (6 months)
    scan3 = ScanRecord(
        scan_id="SCAN_003",
        scan_date=(base_date + timedelta(days=180)).isoformat(),
        disease_state="benign",
        disease_probabilities={"normal": 0.15, "benign": 0.65, "grade_1": 0.15, "grade_2": 0.03, "grade_3": 0.02},
        risk_scores={"1_year": 0.20, "5_year": 0.30, "10_year": 0.35},
        model_version="v1.0.0"
    )
    timeline.add_scan(scan3)
    print(f"  Added scan 3: {scan3.disease_state} (confidence: {scan3.disease_probabilities[scan3.disease_state]:.2%})")
    
    # 3. Initialize longitudinal tracker
    print("\n3. Analyzing disease progression...")
    tracker = LongitudinalTracker(storage_path="patient_timelines")
    tracker.register_timeline(timeline)
    
    # Compute progression metrics
    progression = tracker.compute_progression_metrics(patient_id)
    print(f"\n  Progression Analysis:")
    print(f"    Total scans: {progression['num_scans']}")
    print(f"    Time span: {progression['time_span_days']} days")
    print(f"    Disease state changes: {progression['disease_state_changes']}")
    print(f"    Overall trend: {progression['overall_trend']}")
    
    # 4. Analyze treatment response
    print("\n4. Analyzing treatment response...")
    response_analyzer = TreatmentResponseAnalyzer()
    
    # Compare pre-treatment (scan1) and post-treatment (scan2)
    response = response_analyzer.analyze_treatment_response(
        pre_treatment_state="grade_2",
        post_treatment_state="grade_1",
        pre_treatment_probs=scan1.disease_probabilities,
        post_treatment_probs=scan2.disease_probabilities,
        treatment_type="chemotherapy",
        time_since_treatment_days=83
    )
    
    print(f"\n  Treatment Response:")
    print(f"    Category: {response['response_category']}")
    print(f"    Improvement score: {response['improvement_score']:.3f}")
    print(f"    Probability change: {response['probability_change']:.3f}")
    
    if response['response_category'] == TreatmentResponseCategory.PARTIAL_RESPONSE:
        print(f"    ✓ Partial response to treatment - continue monitoring")
    elif response['response_category'] == TreatmentResponseCategory.COMPLETE_RESPONSE:
        print(f"    ✓✓ Complete response to treatment!")
    
    # 5. Identify significant changes
    print("\n5. Identifying significant changes...")
    changes = tracker.highlight_significant_changes(patient_id, scan3.scan_id)
    
    if changes['significant_changes']:
        print(f"\n  Significant Changes Detected:")
        for change in changes['changes']:
            print(f"    - {change['type']}: {change['description']}")
            print(f"      Magnitude: {change['magnitude']:.3f}")
    else:
        print(f"\n  No significant changes detected")
    
    # 6. Calculate risk evolution
    print("\n6. Analyzing risk evolution...")
    risk_evolution = tracker.calculate_risk_evolution(patient_id)
    
    print(f"\n  Risk Score Trends:")
    for horizon in ["1_year", "5_year", "10_year"]:
        if horizon in risk_evolution:
            trend = risk_evolution[horizon]
            print(f"    {horizon}: {trend['initial']:.2%} → {trend['current']:.2%} (change: {trend['change']:.2%})")
    
    # 7. Save and visualize timeline
    print("\n7. Saving timeline...")
    timeline_path = timeline.save("patient_timeline.json")
    print(f"  Timeline saved to: {timeline_path}")
    
    # Visualize (if matplotlib available)
    try:
        visualizer = TimelineVisualizer()
        fig = visualizer.plot_patient_timeline(timeline)
        fig.savefig("patient_timeline.png", dpi=150, bbox_inches='tight')
        print(f"  Timeline visualization saved to: patient_timeline.png")
    except ImportError:
        print(f"  (Visualization skipped - matplotlib not available)")
    
    # 8. Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Patient: {patient_id}")
    print(f"  Timeline duration: {timeline.get_duration_days()} days")
    print(f"  Total scans: {len(timeline.scans)}")
    print(f"  Treatments: {len(timeline.treatments)}")
    print(f"  Disease progression: {scan1.disease_state} → {scan2.disease_state} → {scan3.disease_state}")
    print(f"  Treatment response: {response['response_category']}")
    print(f"  Current risk (1-year): {scan3.risk_scores['1_year']:.2%}")
    print("\n✓ Longitudinal analysis complete!")


if __name__ == "__main__":
    main()
