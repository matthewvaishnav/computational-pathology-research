"""
Demonstration of longitudinal patient tracking functionality.

This script demonstrates:
1. Creating patient timelines with scans and treatments
2. Computing disease progression metrics
3. Identifying treatment response
4. Calculating risk factor evolution
5. Highlighting significant changes
6. Visualizing patient timelines
"""

import torch
from datetime import datetime, timedelta

from src.clinical.longitudinal import (
    PatientTimeline,
    ScanRecord,
    TreatmentEvent,
    LongitudinalTracker,
)
from src.clinical.patient_context import ClinicalMetadata, SmokingStatus, Sex
from src.clinical.taxonomy import DiseaseTaxonomy
from src.visualization.timeline import TimelineVisualizer


def create_sample_taxonomy():
    """Create a sample disease taxonomy for demonstration."""
    config = {
        "name": "Cancer Grading System",
        "version": "1.0",
        "description": "Hierarchical cancer grading taxonomy",
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
                "children": ["grade_2"],
            },
            {
                "id": "grade_2",
                "name": "Grade 2 Cancer",
                "description": "Moderately differentiated cancer",
                "parent": "grade_1",
                "children": ["grade_3"],
            },
            {
                "id": "grade_3",
                "name": "Grade 3 Cancer",
                "description": "Poorly differentiated cancer",
                "parent": "grade_2",
                "children": [],
            },
        ],
    }
    return DiseaseTaxonomy(config_dict=config)


def create_sample_timeline():
    """Create a sample patient timeline with multiple scans and treatments."""
    timeline = PatientTimeline(patient_id="PATIENT_12345")

    base_date = datetime(2024, 1, 1)

    # Initial scan - Grade 2 cancer detected
    scan1 = ScanRecord(
        scan_id="SCAN_001",
        scan_date=base_date,
        disease_state="grade_2",
        disease_probabilities={"benign": 0.05, "grade_1": 0.15, "grade_2": 0.65, "grade_3": 0.15},
        confidence=0.75,
        risk_scores={
            "grade_3": {"1-year": 0.4, "5-year": 0.6, "10-year": 0.7}
        },
        anomaly_scores={"grade_3": 0.35},
        clinical_metadata=ClinicalMetadata(
            age=62,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.FORMER,
            family_history=["cancer"],
        ),
    )
    timeline.add_scan(scan1)

    # Treatment initiated
    treatment1 = TreatmentEvent(
        treatment_id="TX_001",
        treatment_date=base_date + timedelta(days=14),
        treatment_type="chemotherapy",
        treatment_details={
            "drug": "cisplatin",
            "dose": "75mg/m2",
            "cycles": 6,
        },
    )
    timeline.add_treatment(treatment1)

    # Follow-up scan after 3 months - Partial response
    scan2 = ScanRecord(
        scan_id="SCAN_002",
        scan_date=base_date + timedelta(days=90),
        disease_state="grade_1",
        disease_probabilities={"benign": 0.15, "grade_1": 0.70, "grade_2": 0.12, "grade_3": 0.03},
        confidence=0.82,
        risk_scores={
            "grade_3": {"1-year": 0.25, "5-year": 0.40, "10-year": 0.50}
        },
        anomaly_scores={"grade_3": 0.20},
        clinical_metadata=ClinicalMetadata(
            age=62,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.NEVER,  # Quit smoking
            family_history=["cancer"],
        ),
    )
    timeline.add_scan(scan2)

    # Follow-up scan after 6 months - Continued improvement
    scan3 = ScanRecord(
        scan_id="SCAN_003",
        scan_date=base_date + timedelta(days=180),
        disease_state="benign",
        disease_probabilities={"benign": 0.85, "grade_1": 0.12, "grade_2": 0.02, "grade_3": 0.01},
        confidence=0.90,
        risk_scores={
            "grade_3": {"1-year": 0.10, "5-year": 0.20, "10-year": 0.30}
        },
        anomaly_scores={"grade_3": 0.08},
        clinical_metadata=ClinicalMetadata(
            age=62,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.NEVER,
            family_history=["cancer"],
        ),
    )
    timeline.add_scan(scan3)

    return timeline


def main():
    """Run longitudinal tracking demonstration."""
    print("=" * 80)
    print("Longitudinal Patient Tracking Demonstration")
    print("=" * 80)

    # Create taxonomy
    print("\n1. Creating disease taxonomy...")
    taxonomy = create_sample_taxonomy()
    print(f"   Taxonomy: {taxonomy.name}")
    print(f"   Diseases: {', '.join(taxonomy.disease_ids)}")

    # Create patient timeline
    print("\n2. Creating patient timeline...")
    timeline = create_sample_timeline()
    print(f"   Patient Hash: {timeline.patient_id_hash[:16]}...")
    print(f"   Number of scans: {timeline.get_num_scans()}")
    print(f"   Number of treatments: {timeline.get_num_treatments()}")
    print(f"   Timeline duration: {timeline.get_timeline_duration():.1f} days")

    # Create longitudinal tracker
    print("\n3. Initializing longitudinal tracker...")
    tracker = LongitudinalTracker(taxonomy)
    tracker.register_timeline(timeline)
    print(f"   Registered {tracker.get_num_patients()} patient(s)")

    # Compute progression metrics
    print("\n4. Computing disease progression metrics...")
    progression_metrics = tracker.compute_progression_metrics(timeline)
    print(f"   Overall trend: {progression_metrics['overall_trend']}")
    print(f"   Number of progression events: {len(progression_metrics['progression_events'])}")
    print(f"   Disease state trajectory: {' → '.join(progression_metrics['disease_state_trajectory'])}")
    
    if progression_metrics['progression_events']:
        print("\n   Progression events:")
        for event in progression_metrics['progression_events']:
            print(f"     - Scan {event['scan_id']}: {event['previous_state']} → {event['current_state']}")
            print(f"       Days since previous: {event['days_since_previous']}")
            print(f"       Confidence: {event['confidence']:.2f}")

    # Identify treatment response
    print("\n5. Analyzing treatment response...")
    treatment_response = tracker.identify_treatment_response(timeline, "TX_001")
    print(f"   Treatment: {treatment_response['treatment'].treatment_type}")
    print(f"   Response category: {treatment_response['response_category']}")
    
    if treatment_response['disease_state_change']:
        change = treatment_response['disease_state_change']
        print(f"   Disease state change: {change['from']} → {change['to']}")
    
    if treatment_response['probability_change'] is not None:
        print(f"   Probability change: {treatment_response['probability_change']:+.2f}")
    
    if treatment_response['days_to_response'] is not None:
        print(f"   Days to response: {treatment_response['days_to_response']}")

    # Calculate risk evolution
    print("\n6. Calculating risk factor evolution...")
    risk_evolution = tracker.calculate_risk_evolution(timeline, "grade_3")
    print(f"   Disease: {risk_evolution['disease_id']}")
    print(f"   Risk trend: {risk_evolution['risk_trend']}")
    print(f"   Number of significant changes: {len(risk_evolution['significant_changes'])}")
    
    if risk_evolution['significant_changes']:
        print("\n   Significant risk changes:")
        for change in risk_evolution['significant_changes']:
            print(f"     - Scan {change['scan_id']} ({change['time_horizon']})")
            print(f"       Risk change: {change['previous_risk']:.2f} → {change['current_risk']:.2f} ({change['direction']})")

    # Highlight significant changes for latest scan
    print("\n7. Highlighting significant changes in latest scan...")
    latest_scan = timeline.get_latest_scan()
    changes = tracker.highlight_significant_changes(timeline, latest_scan)
    print(f"   Has significant changes: {changes['has_significant_changes']}")
    print(f"   Disease state changed: {changes['disease_state_changed']}")
    
    if changes['disease_state_change']:
        change = changes['disease_state_change']
        print(f"   Disease state: {change['from']} → {change['to']}")
    
    if changes['confidence_change'] is not None:
        print(f"   Confidence change: {changes['confidence_change']:+.2f}")
    
    print("\n   Clinical recommendations:")
    for rec in changes['recommendations']:
        print(f"     - {rec}")

    # Visualize timeline
    print("\n8. Generating timeline visualization...")
    try:
        visualizer = TimelineVisualizer()
        
        # Create comprehensive timeline plot
        fig = visualizer.plot_timeline(
            timeline,
            disease_ids=["grade_3"],
            show_risk_scores=True,
            show_treatments=True,
            title="Patient Disease Progression Timeline"
        )
        output_path = "patient_timeline.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   Saved timeline visualization to: {output_path}")
        
        # Create progression summary
        fig2 = visualizer.plot_progression_summary(
            progression_metrics,
            title="Disease Progression Summary"
        )
        output_path2 = "progression_summary.png"
        fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
        print(f"   Saved progression summary to: {output_path2}")
        
        # Create treatment response visualization
        fig3 = visualizer.plot_treatment_response(
            treatment_response,
            title="Treatment Response Analysis"
        )
        output_path3 = "treatment_response.png"
        fig3.savefig(output_path3, dpi=150, bbox_inches="tight")
        print(f"   Saved treatment response visualization to: {output_path3}")
        
    except Exception as e:
        print(f"   Warning: Could not generate visualizations: {e}")

    # Save timeline to file
    print("\n9. Saving patient timeline...")
    timeline_path = "patient_timeline.json"
    timeline.save(timeline_path)
    print(f"   Saved timeline to: {timeline_path}")

    # Load timeline from file
    print("\n10. Loading patient timeline...")
    loaded_timeline = PatientTimeline.load(timeline_path)
    print(f"   Loaded timeline with {loaded_timeline.get_num_scans()} scans")

    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
