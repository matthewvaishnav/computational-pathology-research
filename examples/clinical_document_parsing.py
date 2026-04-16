"""
Example demonstrating clinical document parsing for unstructured clinical data.

This example shows how to use the ClinicalDocumentParser to extract structured
information from clinical notes, including diagnoses, medications, procedures,
and observations, with negation and uncertainty detection.
"""

from src.clinical import ClinicalDocumentParser, ClinicalMetadata, SmokingStatus, Sex


def main():
    """Demonstrate clinical document parsing."""
    # Initialize parser
    parser = ClinicalDocumentParser()

    # Example clinical note
    clinical_note = """
    Patient History:
    
    Chief Complaint: Chest pain and shortness of breath
    
    Diagnosis: Hypertension, diabetes mellitus type 2. No evidence of acute myocardial infarction.
    Possible early signs of coronary artery disease.
    
    Medications: Patient is currently taking metformin 500mg twice daily, lisinopril 10mg daily,
    and atorvastatin 20mg at bedtime. Patient denies taking aspirin.
    
    Procedure: Underwent cardiac catheterization on 2024-01-15. Status post coronary angiography.
    
    Findings: Echocardiogram shows mild left ventricular hypertrophy. No significant valvular disease.
    Blood pressure elevated at 145/92 mmHg.
    
    Assessment: Patient has well-controlled diabetes but hypertension requires adjustment.
    Cannot rule out underlying coronary artery disease. Recommend stress test.
    """

    # Parse the clinical note
    print("=" * 80)
    print("CLINICAL DOCUMENT PARSING EXAMPLE")
    print("=" * 80)
    print("\nParsing clinical note...\n")

    parsed = parser.parse_text(clinical_note)

    # Display extracted diagnoses
    print("DIAGNOSES:")
    print("-" * 80)
    for diagnosis in parsed.diagnoses:
        status = []
        if diagnosis.negated:
            status.append("NEGATED")
        if diagnosis.uncertain:
            status.append("UNCERTAIN")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"  • {diagnosis.value}")
        print(
            f"    Confidence: {diagnosis.confidence:.2f} ({diagnosis.get_confidence_level().value}){status_str}"
        )
        print()

    # Display extracted medications
    print("\nMEDICATIONS:")
    print("-" * 80)
    for medication in parsed.medications:
        status = []
        if medication.negated:
            status.append("NEGATED")
        if medication.uncertain:
            status.append("UNCERTAIN")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"  • {medication.value}")
        print(
            f"    Confidence: {medication.confidence:.2f} ({medication.get_confidence_level().value}){status_str}"
        )
        print()

    # Display extracted procedures
    print("\nPROCEDURES:")
    print("-" * 80)
    for procedure in parsed.procedures:
        status = []
        if procedure.negated:
            status.append("NEGATED")
        if procedure.uncertain:
            status.append("UNCERTAIN")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"  • {procedure.value}")
        print(
            f"    Confidence: {procedure.confidence:.2f} ({procedure.get_confidence_level().value}){status_str}"
        )
        print()

    # Display extracted observations
    print("\nOBSERVATIONS:")
    print("-" * 80)
    for observation in parsed.observations:
        status = []
        if observation.negated:
            status.append("NEGATED")
        if observation.uncertain:
            status.append("UNCERTAIN")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"  • {observation.value}")
        print(
            f"    Confidence: {observation.confidence:.2f} ({observation.get_confidence_level().value}){status_str}"
        )
        print()

    # Check for conflicts with structured metadata
    print("\nCONFLICT DETECTION:")
    print("-" * 80)

    # Simulate structured metadata
    structured_metadata = {
        "medications": ["metformin", "lisinopril", "atorvastatin", "aspirin"],
        "diagnoses": ["hypertension", "diabetes mellitus"],
    }

    conflicts = parser.check_conflicts(parsed, structured_metadata)

    if conflicts:
        print("Conflicts detected:")
        for conflict in conflicts:
            print(f"  WARNING: {conflict}")
    else:
        print("No conflicts detected between document and structured metadata.")

    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total entities extracted: {len(parsed.get_all_entities())}")
    print(f"High-confidence entities: {len(parsed.get_high_confidence_entities())}")
    print(f"Diagnoses: {len(parsed.diagnoses)}")
    print(f"Medications: {len(parsed.medications)}")
    print(f"Procedures: {len(parsed.procedures)}")
    print(f"Observations: {len(parsed.observations)}")

    # Count negated and uncertain entities
    all_entities = parsed.get_all_entities()
    negated_count = sum(1 for e in all_entities if e.negated)
    uncertain_count = sum(1 for e in all_entities if e.uncertain)

    print(f"Negated entities: {negated_count}")
    print(f"Uncertain entities: {uncertain_count}")

    # Example: Integration with ClinicalMetadata
    print("\n" + "=" * 80)
    print("INTEGRATION WITH CLINICAL METADATA")
    print("=" * 80)

    # Create clinical metadata from structured data
    metadata = ClinicalMetadata(
        age=65,
        sex=Sex.MALE,
        smoking_status=SmokingStatus.FORMER,
        medications=["metformin", "lisinopril", "atorvastatin"],
    )

    print("\nStructured Clinical Metadata:")
    print(f"  Age: {metadata.age}")
    print(f"  Sex: {metadata.sex.value}")
    print(f"  Smoking Status: {metadata.smoking_status.value}")
    print(f"  Medications: {', '.join(metadata.medications)}")

    print("\nExtracted from Document:")
    extracted_meds = [m.value for m in parsed.medications if not m.negated]
    print(f"  Medications: {', '.join(extracted_meds)}")

    print("\n" + "=" * 80)
    print("Document parsing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
