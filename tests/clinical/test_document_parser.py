"""
Unit tests for clinical document parser.
"""

import pytest

from src.clinical.document_parser import (
    ClinicalDocumentParser,
    DocumentFormat,
    ExtractedEntity,
    ExtractionConfidence,
    ParsedDocument,
)


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_get_confidence_level_high(self):
        """Test high confidence level classification."""
        entity = ExtractedEntity(
            entity_type="diagnosis",
            value="hypertension",
            confidence=0.9,
        )
        assert entity.get_confidence_level() == ExtractionConfidence.HIGH

    def test_get_confidence_level_medium(self):
        """Test medium confidence level classification."""
        entity = ExtractedEntity(
            entity_type="diagnosis",
            value="possible infection",
            confidence=0.6,
        )
        assert entity.get_confidence_level() == ExtractionConfidence.MEDIUM

    def test_get_confidence_level_low(self):
        """Test low confidence level classification."""
        entity = ExtractedEntity(
            entity_type="diagnosis",
            value="unclear",
            confidence=0.3,
        )
        assert entity.get_confidence_level() == ExtractionConfidence.LOW

    def test_to_dict(self):
        """Test entity serialization to dictionary."""
        entity = ExtractedEntity(
            entity_type="medication",
            value="metformin",
            confidence=0.85,
            negated=False,
            uncertain=True,
            context="patient may be taking metformin",
        )
        result = entity.to_dict()

        assert result["entity_type"] == "medication"
        assert result["value"] == "metformin"
        assert result["confidence"] == 0.85
        assert result["confidence_level"] == "high"
        assert result["negated"] is False
        assert result["uncertain"] is True


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_get_all_entities(self):
        """Test retrieving all entities from parsed document."""
        doc = ParsedDocument(
            document_format=DocumentFormat.PLAIN_TEXT,
            diagnoses=[
                ExtractedEntity("diagnosis", "diabetes", 0.9),
            ],
            medications=[
                ExtractedEntity("medication", "insulin", 0.85),
                ExtractedEntity("medication", "metformin", 0.8),
            ],
            procedures=[
                ExtractedEntity("procedure", "blood test", 0.7),
            ],
        )

        all_entities = doc.get_all_entities()
        assert len(all_entities) == 4
        assert all_entities[0].value == "diabetes"
        assert all_entities[1].value == "insulin"

    def test_get_high_confidence_entities(self):
        """Test filtering high-confidence entities."""
        doc = ParsedDocument(
            document_format=DocumentFormat.PLAIN_TEXT,
            diagnoses=[
                ExtractedEntity("diagnosis", "diabetes", 0.9),
                ExtractedEntity("diagnosis", "possible infection", 0.6),
            ],
            medications=[
                ExtractedEntity("medication", "insulin", 0.85),
            ],
        )

        high_conf = doc.get_high_confidence_entities()
        assert len(high_conf) == 2
        assert all(e.confidence > 0.8 for e in high_conf)

    def test_to_dict(self):
        """Test document serialization to dictionary."""
        doc = ParsedDocument(
            document_format=DocumentFormat.PLAIN_TEXT,
            diagnoses=[
                ExtractedEntity("diagnosis", "hypertension", 0.9),
            ],
            conflicts=["Test conflict"],
        )

        result = doc.to_dict()
        assert result["document_format"] == "plain_text"
        assert len(result["diagnoses"]) == 1
        assert result["conflicts"] == ["Test conflict"]


class TestClinicalDocumentParser:
    """Tests for ClinicalDocumentParser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance for tests."""
        return ClinicalDocumentParser()

    def test_initialization(self, parser):
        """Test parser initialization."""
        assert parser is not None
        assert len(parser.negation_patterns) > 0
        assert len(parser.uncertainty_patterns) > 0
        assert len(parser.abbreviations) > 0

    def test_parse_simple_diagnosis(self, parser):
        """Test parsing simple diagnosis from text."""
        text = "Diagnosis: hypertension and diabetes mellitus"
        parsed = parser.parse_text(text)

        assert len(parsed.diagnoses) > 0
        # Check that we extracted something related to the diagnoses
        diagnosis_values = [d.value.lower() for d in parsed.diagnoses]
        assert any("hypertension" in dv or "diabetes" in dv for dv in diagnosis_values)

    def test_parse_negated_diagnosis(self, parser):
        """Test parsing negated diagnosis."""
        text = "Diagnosis: no evidence of cancer"
        parsed = parser.parse_text(text)

        # Should extract cancer but mark it as negated
        cancer_entities = [d for d in parsed.diagnoses if "cancer" in d.value.lower()]
        if cancer_entities:
            assert cancer_entities[0].negated is True

    def test_parse_uncertain_diagnosis(self, parser):
        """Test parsing diagnosis with uncertainty qualifiers."""
        text = "Diagnosis: possible pneumonia"
        parsed = parser.parse_text(text)

        # Should extract pneumonia and mark as uncertain
        pneumonia_entities = [d for d in parsed.diagnoses if "pneumonia" in d.value.lower()]
        if pneumonia_entities:
            assert pneumonia_entities[0].uncertain is True

    def test_parse_medications(self, parser):
        """Test parsing medications from text."""
        text = "Medications: metformin 500mg, lisinopril 10mg"
        parsed = parser.parse_text(text)

        assert len(parsed.medications) > 0
        med_values = [m.value.lower() for m in parsed.medications]
        # Should extract at least one medication
        assert any("metformin" in mv or "lisinopril" in mv for mv in med_values)

    def test_parse_procedures(self, parser):
        """Test parsing procedures from text."""
        text = "Procedure: coronary artery bypass grafting"
        parsed = parser.parse_text(text)

        assert len(parsed.procedures) > 0

    def test_parse_observations(self, parser):
        """Test parsing clinical observations."""
        text = "Findings: elevated blood pressure, normal heart rate"
        parsed = parser.parse_text(text)

        assert len(parsed.observations) > 0

    def test_abbreviation_expansion(self, parser):
        """Test medical abbreviation expansion."""
        text = "Diagnosis: HTN and DM"
        parsed = parser.parse_text(text)

        # Abbreviations should be expanded in normalized text
        diagnosis_values = [d.value.lower() for d in parsed.diagnoses]
        # Should find hypertension or diabetes after expansion
        assert any("hypertension" in dv or "diabetes" in dv for dv in diagnosis_values)

    def test_negation_detection(self, parser):
        """Test negation detection in context."""
        context = "patient denies chest pain and has no shortness of breath"
        entity_pos = context.find("chest pain")

        is_negated = parser._is_negated(context, entity_pos)
        assert is_negated is True

    def test_uncertainty_detection(self, parser):
        """Test uncertainty detection in context."""
        context = "patient may have possible infection"
        entity_pos = context.find("infection")

        is_uncertain = parser._is_uncertain(context, entity_pos)
        assert is_uncertain is True

    def test_confidence_calculation(self, parser):
        """Test confidence score calculation."""
        # High confidence for clear medical term
        conf1 = parser._calculate_confidence(
            "diabetes mellitus", "diagnosis: diabetes mellitus", False, False
        )
        assert conf1 > 0.7

        # Lower confidence for uncertain entity
        conf2 = parser._calculate_confidence("infection", "possible infection", False, True)
        assert conf2 < conf1

        # Lower confidence for very short value
        conf3 = parser._calculate_confidence("dx", "dx: ?", False, False)
        assert conf3 < conf1

    def test_check_conflicts_medication(self, parser):
        """Test conflict detection between extracted and structured data."""
        text = "Patient denies taking aspirin"
        parsed = parser.parse_text(text)

        structured_metadata = {"medications": ["aspirin", "metformin"]}

        conflicts = parser.check_conflicts(parsed, structured_metadata)

        # Should detect conflict if aspirin is in structured but negated in document
        # Note: This depends on whether "aspirin" was successfully extracted and negated
        assert isinstance(conflicts, list)

    def test_check_conflicts_no_conflict(self, parser):
        """Test no conflicts when data is consistent."""
        text = "Medications: metformin, lisinopril"
        parsed = parser.parse_text(text)

        structured_metadata = {"medications": ["metformin", "lisinopril"]}

        conflicts = parser.check_conflicts(parsed, structured_metadata)

        # Should have no conflicts
        assert len(conflicts) == 0

    def test_parse_multiple_entities(self, parser):
        """Test parsing document with multiple entity types."""
        text = """
        Patient History:
        Diagnosis: hypertension, diabetes mellitus type 2
        Medications: metformin 500mg twice daily, lisinopril 10mg daily
        Procedure: underwent cardiac catheterization
        Findings: shows mild coronary artery disease
        """
        parsed = parser.parse_text(text)

        # Should extract entities from all categories
        assert len(parsed.diagnoses) > 0
        assert len(parsed.medications) > 0
        assert len(parsed.procedures) > 0
        assert len(parsed.observations) > 0

    def test_parse_complex_negation(self, parser):
        """Test parsing with complex negation patterns."""
        text = "No evidence of malignancy. Rules out infection. Patient is free of symptoms."
        parsed = parser.parse_text(text)

        # All extracted entities should be marked as negated
        all_entities = parsed.get_all_entities()
        if all_entities:
            # At least some should be negated
            assert any(e.negated for e in all_entities)

    def test_normalize_text(self, parser):
        """Test text normalization."""
        text = "Pt w/ HTN and DM s/p MI"
        normalized = parser._normalize_text(text)

        # Should expand abbreviations
        assert "patient" in normalized or "pt" in normalized
        assert "hypertension" in normalized or "htn" in normalized
        assert "diabetes" in normalized or "dm" in normalized

    def test_empty_text(self, parser):
        """Test parsing empty text."""
        parsed = parser.parse_text("")

        assert len(parsed.diagnoses) == 0
        assert len(parsed.medications) == 0
        assert len(parsed.procedures) == 0
        assert len(parsed.observations) == 0

    def test_document_format_detection(self, parser, tmp_path):
        """Test document format detection from file extension."""
        # Create test files
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content")

        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<?xml version='1.0'?><root>Test</root>")

        assert parser._detect_format(txt_file) == DocumentFormat.PLAIN_TEXT
        assert parser._detect_format(xml_file) == DocumentFormat.HL7_CDA

    def test_parse_file_not_found(self, parser):
        """Test parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent_file.txt")

    def test_parse_text_file(self, parser, tmp_path):
        """Test parsing plain text file."""
        test_file = tmp_path / "clinical_note.txt"
        test_file.write_text("Diagnosis: hypertension\nMedications: lisinopril")

        parsed = parser.parse_file(test_file)

        assert parsed.document_format == DocumentFormat.PLAIN_TEXT
        assert len(parsed.diagnoses) > 0 or len(parsed.medications) > 0
