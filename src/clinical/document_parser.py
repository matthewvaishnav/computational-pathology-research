"""
Clinical document parser for extracting structured information from unstructured clinical data.

This module provides parsing capabilities for common clinical document formats (HL7 CDA, plain text, PDF)
and extracts structured information including diagnoses, medications, procedures, and clinical observations.
It handles medical abbreviations, terminology variations, negation detection, and uncertainty qualifiers.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class DocumentFormat(str, Enum):
    """Supported clinical document formats."""

    PLAIN_TEXT = "plain_text"
    HL7_CDA = "hl7_cda"
    PDF = "pdf"
    UNKNOWN = "unknown"


class ExtractionConfidence(str, Enum):
    """Confidence levels for extracted information."""

    HIGH = "high"  # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # <0.5


@dataclass
class ExtractedEntity:
    """
    Represents an extracted clinical entity from unstructured text.

    Attributes:
        entity_type: Type of entity (diagnosis, medication, procedure, observation)
        value: Extracted value/text
        confidence: Confidence score (0.0-1.0)
        negated: Whether the entity is negated (e.g., "no evidence of cancer")
        uncertain: Whether the entity has uncertainty qualifiers (e.g., "possible", "suspected")
        context: Surrounding text context for verification
        source_location: Location in source document (line number, character offset)
    """

    entity_type: str
    value: str
    confidence: float
    negated: bool = False
    uncertain: bool = False
    context: str = ""
    source_location: Optional[str] = None

    def get_confidence_level(self) -> ExtractionConfidence:
        """Get categorical confidence level."""
        if self.confidence > 0.8:
            return ExtractionConfidence.HIGH
        elif self.confidence >= 0.5:
            return ExtractionConfidence.MEDIUM
        else:
            return ExtractionConfidence.LOW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "confidence": self.confidence,
            "confidence_level": self.get_confidence_level().value,
            "negated": self.negated,
            "uncertain": self.uncertain,
            "context": self.context,
            "source_location": self.source_location,
        }


@dataclass
class ParsedDocument:
    """
    Represents a parsed clinical document with extracted structured information.

    Attributes:
        document_format: Format of the source document
        diagnoses: Extracted diagnoses
        medications: Extracted medications
        procedures: Extracted procedures
        observations: Extracted clinical observations
        conflicts: List of conflicts with structured metadata
        raw_text: Original document text (optional)
    """

    document_format: DocumentFormat
    diagnoses: List[ExtractedEntity] = field(default_factory=list)
    medications: List[ExtractedEntity] = field(default_factory=list)
    procedures: List[ExtractedEntity] = field(default_factory=list)
    observations: List[ExtractedEntity] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    raw_text: Optional[str] = None

    def get_all_entities(self) -> List[ExtractedEntity]:
        """Get all extracted entities across all categories."""
        return (
            self.diagnoses + self.medications + self.procedures + self.observations
        )

    def get_high_confidence_entities(self) -> List[ExtractedEntity]:
        """Get only high-confidence extracted entities."""
        return [
            entity
            for entity in self.get_all_entities()
            if entity.get_confidence_level() == ExtractionConfidence.HIGH
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_format": self.document_format.value,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "medications": [m.to_dict() for m in self.medications],
            "procedures": [p.to_dict() for p in self.procedures],
            "observations": [o.to_dict() for o in self.observations],
            "conflicts": self.conflicts,
        }


class ClinicalDocumentParser:
    """
    Parser for extracting structured information from unstructured clinical documents.

    This parser supports multiple document formats (HL7 CDA, plain text, PDF) and extracts
    structured clinical information including diagnoses, medications, procedures, and observations.
    It handles medical abbreviations, terminology variations, negation detection, and uncertainty
    qualifiers to provide accurate semantic interpretation.

    Example:
        >>> parser = ClinicalDocumentParser()
        >>> parsed = parser.parse_text(clinical_note)
        >>> for diagnosis in parsed.diagnoses:
        ...     print(f"{diagnosis.value} (confidence: {diagnosis.confidence:.2f}, negated: {diagnosis.negated})")
    """

    def __init__(self):
        """Initialize the clinical document parser."""
        # Negation patterns (words/phrases indicating negation)
        self.negation_patterns = [
            r"\bno\b",
            r"\bnot\b",
            r"\bdenies\b",
            r"\bdenied\b",
            r"\bwithout\b",
            r"\babsent\b",
            r"\babsence of\b",
            r"\bno evidence of\b",
            r"\bno signs of\b",
            r"\bno symptoms of\b",
            r"\bnegative for\b",
            r"\brules out\b",
            r"\bruled out\b",
            r"\bfree of\b",
        ]

        # Uncertainty patterns (words/phrases indicating uncertainty)
        self.uncertainty_patterns = [
            r"\bpossible\b",
            r"\bpossibly\b",
            r"\bprobable\b",
            r"\bprobably\b",
            r"\bsuspected\b",
            r"\bsuspect\b",
            r"\blikely\b",
            r"\bunlikely\b",
            r"\bmay\b",
            r"\bmight\b",
            r"\bcould\b",
            r"\bquestionable\b",
            r"\buncertain\b",
            r"\bunclear\b",
            r"\bsuggests\b",
            r"\bsuggesting\b",
            r"\bconsistent with\b",
            r"\bcompatible with\b",
            r"\bcannot rule out\b",
            r"\bcannot exclude\b",
        ]

        # Medical abbreviations mapping (common abbreviations to full terms)
        self.abbreviations = {
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "mi": "myocardial infarction",
            "ca": "cancer",
            "hx": "history",
            "dx": "diagnosis",
            "tx": "treatment",
            "rx": "prescription",
            "pt": "patient",
            "w/": "with",
            "w/o": "without",
            "s/p": "status post",
            "r/o": "rule out",
            "nkda": "no known drug allergies",
            "sob": "shortness of breath",
            "cp": "chest pain",
            "n/v": "nausea and vomiting",
            "abd": "abdominal",
            "wnl": "within normal limits",
        }

        # Clinical entity patterns (regex patterns for identifying clinical entities)
        self.diagnosis_patterns = [
            r"(?:diagnosis|diagnosed with|dx):\s*([^.\n]+)",
            r"(?:impression|assessment):\s*([^.\n]+)",
            r"(?:condition|disease):\s*([^.\n]+)",
        ]

        self.medication_patterns = [
            r"(?:medication|medications|rx|prescribed):\s*([^.\n]+)",
            r"(?:taking|on)\s+([a-z]+(?:in|ol|ide|pril|sartan|statin)\b)",
            r"\b([a-z]+(?:in|ol|ide|pril|sartan|statin))\s+\d+\s*mg\b",
        ]

        self.procedure_patterns = [
            r"(?:procedure|surgery|operation):\s*([^.\n]+)",
            r"(?:underwent|performed)\s+([^.\n]+)",
            r"s/p\s+([^.\n]+)",
        ]

        self.observation_patterns = [
            r"(?:findings|observation|noted):\s*([^.\n]+)",
            r"(?:shows|demonstrates|reveals)\s+([^.\n]+)",
        ]

        logger.info("Initialized ClinicalDocumentParser")

    def parse_file(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a clinical document file.

        Args:
            file_path: Path to the clinical document file

        Returns:
            ParsedDocument with extracted structured information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        # Detect document format
        document_format = self._detect_format(file_path)

        # Read document content
        if document_format == DocumentFormat.PLAIN_TEXT:
            text = self._read_text_file(file_path)
        elif document_format == DocumentFormat.HL7_CDA:
            text = self._parse_hl7_cda(file_path)
        elif document_format == DocumentFormat.PDF:
            text = self._parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported document format: {document_format}")

        # Parse the text content
        return self.parse_text(text, document_format=document_format)

    def parse_text(
        self,
        text: str,
        document_format: DocumentFormat = DocumentFormat.PLAIN_TEXT,
    ) -> ParsedDocument:
        """
        Parse clinical text and extract structured information.

        Args:
            text: Clinical document text
            document_format: Format of the document

        Returns:
            ParsedDocument with extracted structured information
        """
        # Normalize text (expand abbreviations, lowercase for matching)
        normalized_text = self._normalize_text(text)

        # Extract entities
        diagnoses = self._extract_diagnoses(normalized_text, text)
        medications = self._extract_medications(normalized_text, text)
        procedures = self._extract_procedures(normalized_text, text)
        observations = self._extract_observations(normalized_text, text)

        # Create parsed document
        parsed = ParsedDocument(
            document_format=document_format,
            diagnoses=diagnoses,
            medications=medications,
            procedures=procedures,
            observations=observations,
            raw_text=text,
        )

        logger.info(
            f"Parsed document: {len(diagnoses)} diagnoses, {len(medications)} medications, "
            f"{len(procedures)} procedures, {len(observations)} observations"
        )

        return parsed

    def check_conflicts(
        self,
        parsed_document: ParsedDocument,
        structured_metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Check for conflicts between extracted information and structured metadata.

        Args:
            parsed_document: Parsed document with extracted entities
            structured_metadata: Structured clinical metadata (e.g., from ClinicalMetadata)

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Check medication conflicts
        if "medications" in structured_metadata:
            structured_meds = set(
                med.lower() for med in structured_metadata["medications"]
            )
            extracted_meds = set(
                med.value.lower()
                for med in parsed_document.medications
                if not med.negated
            )

            # Find medications in structured but negated in document
            for med in structured_meds:
                negated_in_doc = any(
                    med in entity.value.lower() and entity.negated
                    for entity in parsed_document.medications
                )
                if negated_in_doc:
                    conflicts.append(
                        f"Medication conflict: '{med}' listed in structured data but negated in document"
                    )

        # Check diagnosis conflicts (if structured metadata has diagnoses)
        if "diagnoses" in structured_metadata:
            structured_dx = set(dx.lower() for dx in structured_metadata["diagnoses"])
            extracted_dx = set(
                dx.value.lower()
                for dx in parsed_document.diagnoses
                if not dx.negated
            )

            # Find diagnoses in structured but negated in document
            for dx in structured_dx:
                negated_in_doc = any(
                    dx in entity.value.lower() and entity.negated
                    for entity in parsed_document.diagnoses
                )
                if negated_in_doc:
                    conflicts.append(
                        f"Diagnosis conflict: '{dx}' listed in structured data but negated in document"
                    )

        parsed_document.conflicts = conflicts
        return conflicts

    def _detect_format(self, file_path: Path) -> DocumentFormat:
        """Detect document format from file extension and content."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return DocumentFormat.PDF
        elif suffix in [".xml", ".cda"]:
            return DocumentFormat.HL7_CDA
        elif suffix in [".txt", ".text"]:
            return DocumentFormat.PLAIN_TEXT
        else:
            # Try to detect from content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read(1000)  # Read first 1000 chars
                    if "<?xml" in content or "<ClinicalDocument" in content:
                        return DocumentFormat.HL7_CDA
                    else:
                        return DocumentFormat.PLAIN_TEXT
            except Exception:
                return DocumentFormat.UNKNOWN

    def _read_text_file(self, file_path: Path) -> str:
        """Read plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_hl7_cda(self, file_path: Path) -> str:
        """
        Parse HL7 CDA XML document and extract text content.

        Note: This is a simplified implementation. Production systems should use
        proper HL7 CDA parsing libraries.
        """
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract text from common CDA sections
            text_parts = []

            # Try to find text elements (simplified extraction)
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())

            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Failed to parse HL7 CDA document: {e}")
            # Fallback to reading as text
            return self._read_text_file(file_path)

    def _parse_pdf(self, file_path: Path) -> str:
        """
        Parse PDF document and extract text content.

        Note: This is a placeholder. Production systems should use PDF parsing
        libraries like PyPDF2, pdfplumber, or Apache Tika.
        """
        logger.warning(
            "PDF parsing not fully implemented. Install PyPDF2 or pdfplumber for PDF support."
        )
        raise NotImplementedError(
            "PDF parsing requires additional dependencies (PyPDF2 or pdfplumber)"
        )

    def _normalize_text(self, text: str) -> str:
        """
        Normalize clinical text by expanding abbreviations and standardizing format.

        Args:
            text: Raw clinical text

        Returns:
            Normalized text
        """
        normalized = text.lower()

        # Expand abbreviations
        for abbrev, full_term in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            normalized = re.sub(pattern, full_term, normalized)

        return normalized

    def _extract_diagnoses(
        self, normalized_text: str, original_text: str
    ) -> List[ExtractedEntity]:
        """Extract diagnoses from clinical text."""
        entities = []

        for pattern in self.diagnosis_patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value:
                    # Get context window
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_text), match.end() + 50)
                    context = normalized_text[start:end]

                    # Check for negation and uncertainty
                    negated = self._is_negated(context, match.start() - start)
                    uncertain = self._is_uncertain(context, match.start() - start)

                    # Calculate confidence (simple heuristic)
                    confidence = self._calculate_confidence(
                        value, context, negated, uncertain
                    )

                    entities.append(
                        ExtractedEntity(
                            entity_type="diagnosis",
                            value=value,
                            confidence=confidence,
                            negated=negated,
                            uncertain=uncertain,
                            context=context,
                        )
                    )

        return entities

    def _extract_medications(
        self, normalized_text: str, original_text: str
    ) -> List[ExtractedEntity]:
        """Extract medications from clinical text."""
        entities = []

        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value:
                    # Get context window
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_text), match.end() + 50)
                    context = normalized_text[start:end]

                    # Check for negation and uncertainty
                    negated = self._is_negated(context, match.start() - start)
                    uncertain = self._is_uncertain(context, match.start() - start)

                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        value, context, negated, uncertain
                    )

                    entities.append(
                        ExtractedEntity(
                            entity_type="medication",
                            value=value,
                            confidence=confidence,
                            negated=negated,
                            uncertain=uncertain,
                            context=context,
                        )
                    )

        return entities

    def _extract_procedures(
        self, normalized_text: str, original_text: str
    ) -> List[ExtractedEntity]:
        """Extract procedures from clinical text."""
        entities = []

        for pattern in self.procedure_patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value:
                    # Get context window
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_text), match.end() + 50)
                    context = normalized_text[start:end]

                    # Check for negation and uncertainty
                    negated = self._is_negated(context, match.start() - start)
                    uncertain = self._is_uncertain(context, match.start() - start)

                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        value, context, negated, uncertain
                    )

                    entities.append(
                        ExtractedEntity(
                            entity_type="procedure",
                            value=value,
                            confidence=confidence,
                            negated=negated,
                            uncertain=uncertain,
                            context=context,
                        )
                    )

        return entities

    def _extract_observations(
        self, normalized_text: str, original_text: str
    ) -> List[ExtractedEntity]:
        """Extract clinical observations from text."""
        entities = []

        for pattern in self.observation_patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value:
                    # Get context window
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_text), match.end() + 50)
                    context = normalized_text[start:end]

                    # Check for negation and uncertainty
                    negated = self._is_negated(context, match.start() - start)
                    uncertain = self._is_uncertain(context, match.start() - start)

                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        value, context, negated, uncertain
                    )

                    entities.append(
                        ExtractedEntity(
                            entity_type="observation",
                            value=value,
                            confidence=confidence,
                            negated=negated,
                            uncertain=uncertain,
                            context=context,
                        )
                    )

        return entities

    def _is_negated(self, context: str, entity_position: int) -> bool:
        """
        Check if an entity is negated based on surrounding context.

        Args:
            context: Text context around the entity
            entity_position: Position of entity within context

        Returns:
            True if entity is negated
        """
        # Look for negation patterns in the preceding text (within 5 words before entity)
        preceding_text = context[:entity_position]
        words_before = preceding_text.split()[-5:]  # Last 5 words only
        preceding_window = " ".join(words_before)

        # Also check the context itself for negation patterns (for cases like "no evidence of X")
        for pattern in self.negation_patterns:
            if re.search(pattern, preceding_window, re.IGNORECASE):
                return True
            # Check if negation appears in the broader context window
            if re.search(pattern, context, re.IGNORECASE):
                # Make sure it's close to the entity (within 50 chars before)
                match = re.search(pattern, context, re.IGNORECASE)
                if match and entity_position - match.start() < 50:
                    return True

        return False

    def _is_uncertain(self, context: str, entity_position: int) -> bool:
        """
        Check if an entity has uncertainty qualifiers.

        Args:
            context: Text context around the entity
            entity_position: Position of entity within context

        Returns:
            True if entity has uncertainty qualifiers
        """
        # Look for uncertainty patterns in surrounding text (within 10 words before/after)
        preceding_text = context[:entity_position]
        following_text = context[entity_position:]

        words_before = preceding_text.split()[-10:]
        words_after = following_text.split()[:10]

        surrounding_window = " ".join(words_before + words_after)

        for pattern in self.uncertainty_patterns:
            if re.search(pattern, surrounding_window, re.IGNORECASE):
                return True

        return False

    def _calculate_confidence(
        self, value: str, context: str, negated: bool, uncertain: bool
    ) -> float:
        """
        Calculate confidence score for extracted entity.

        Args:
            value: Extracted entity value
            context: Surrounding context
            negated: Whether entity is negated
            uncertain: Whether entity has uncertainty qualifiers

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with base confidence
        confidence = 0.8

        # Reduce confidence for very short values (likely incomplete)
        if len(value) < 5:
            confidence -= 0.2

        # Reduce confidence for uncertain entities
        if uncertain:
            confidence -= 0.2

        # Negated entities still have high confidence (we're confident they're negated)
        # No reduction for negation

        # Increase confidence if value contains specific medical terms
        medical_terms = [
            "cancer",
            "carcinoma",
            "disease",
            "syndrome",
            "disorder",
            "hypertension",
            "diabetes",
        ]
        if any(term in value.lower() for term in medical_terms):
            confidence += 0.1

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))
