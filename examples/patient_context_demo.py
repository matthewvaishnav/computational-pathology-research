"""
Demo script for patient context integration.

This script demonstrates how to use the patient context integration system
to combine imaging features with clinical metadata for improved diagnostic predictions.
"""

import torch

from src.clinical import (
    AlcoholConsumption,
    ClinicalMetadata,
    ExerciseFrequency,
    PatientContextIntegrator,
    Sex,
    SmokingStatus,
)


def main():
    """Demonstrate patient context integration."""
    print("=" * 70)
    print("Patient Context Integration Demo")
    print("=" * 70)

    # Create clinical metadata for multiple patients
    print("\n1. Creating clinical metadata for patients...")

    patient1 = ClinicalMetadata(
        smoking_status=SmokingStatus.FORMER,
        alcohol_consumption=AlcoholConsumption.LIGHT,
        medications=["metformin", "lisinopril"],
        exercise_frequency=ExerciseFrequency.MODERATE,
        age=65,
        sex=Sex.MALE,
        family_history=["diabetes", "hypertension"],
    )

    patient2 = ClinicalMetadata(
        smoking_status=SmokingStatus.NEVER,
        alcohol_consumption=AlcoholConsumption.NONE,
        medications=["vitamin_d"],
        exercise_frequency=ExerciseFrequency.ACTIVE,
        age=45,
        sex=Sex.FEMALE,
        family_history=[],
    )

    patient3 = ClinicalMetadata(
        smoking_status=SmokingStatus.CURRENT,
        alcohol_consumption=AlcoholConsumption.MODERATE,
        medications=["aspirin", "atorvastatin"],
        exercise_frequency=ExerciseFrequency.LIGHT,
        age=58,
        sex=Sex.MALE,
        family_history=["lung_cancer", "copd"],
    )

    patients = [patient1, patient2, patient3]

    # Display patient information
    for i, patient in enumerate(patients, 1):
        print(f"\nPatient {i}: {patient}")
        print(f"  Complete: {patient.is_complete()}")
        missing = patient.get_missing_fields()
        if missing:
            print(f"  Missing fields: {missing}")

    # Create patient context integrator
    print("\n2. Initializing patient context integrator...")
    integrator = PatientContextIntegrator(
        embed_dim=256,
        modalities=["wsi", "clinical_metadata"],
    )
    print(f"   Modalities: {integrator.modalities}")

    # Simulate WSI features (in practice, these come from MIL models)
    print("\n3. Simulating WSI features...")
    batch_size = len(patients)
    wsi_features = torch.randn(batch_size, 256)
    print(f"   WSI features shape: {wsi_features.shape}")

    # Generate multimodal patient representations
    print("\n4. Generating multimodal patient representations...")
    patient_representations = integrator(
        imaging_features={"wsi": wsi_features},
        clinical_metadata=patients,
    )
    print(f"   Patient representations shape: {patient_representations.shape}")
    print(f"   Contains NaN: {torch.isnan(patient_representations).any()}")

    # Get metadata availability statistics
    print("\n5. Metadata availability statistics:")
    availability = integrator.get_metadata_availability(patients)
    for field, percent in availability.items():
        print(f"   {field}: {percent:.1f}%")

    # Demonstrate handling missing metadata
    print("\n6. Testing with incomplete metadata...")
    incomplete_patient = ClinicalMetadata(age=50)  # Only age provided
    print(f"   Incomplete patient: {incomplete_patient}")
    print(f"   Missing fields: {incomplete_patient.get_missing_fields()}")

    wsi_single = torch.randn(1, 256)
    patient_repr_incomplete = integrator(
        imaging_features={"wsi": wsi_single},
        clinical_metadata=[incomplete_patient],
    )
    print(f"   Generated representation shape: {patient_repr_incomplete.shape}")

    # Demonstrate handling no metadata
    print("\n7. Testing with no metadata (imaging only)...")
    patient_repr_no_metadata = integrator(
        imaging_features={"wsi": wsi_single},
        clinical_metadata=None,
    )
    print(f"   Generated representation shape: {patient_repr_no_metadata.shape}")

    # Demonstrate serialization
    print("\n8. Testing metadata serialization...")
    metadata_dict = patient1.to_dict()
    print(f"   Serialized to dict: {list(metadata_dict.keys())}")

    # Reconstruct from dict
    reconstructed = ClinicalMetadata.from_dict(metadata_dict)
    print(f"   Reconstructed: {reconstructed}")
    print(f"   Matches original: {reconstructed.age == patient1.age}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
