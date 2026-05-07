"""Tests for EMR integration."""

from datetime import datetime

import pytest

from src.streaming.emr_integration import (
    ClinicalNote,
    EMRConfig,
    EMRIntegrationClient,
    EMRIntegrationFactory,
    EMRVendor,
    PatientRecord,
)


class TestEMRIntegrationClient:
    """Tests for EMRIntegrationClient."""

    def test_init(self):
        """Test client initialization."""
        config = EMRConfig(
            vendor=EMRVendor.EPIC,
            base_url="https://emr.example.com",
            client_id="test_client",
            timeout=60,
        )

        client = EMRIntegrationClient(config)

        assert client.vendor == EMRVendor.EPIC
        assert client.config.base_url == "https://emr.example.com"
        assert client.config.timeout == 60

    def test_get_patient_record_not_implemented(self):
        """Test patient record retrieval (not implemented)."""
        config = EMRConfig(vendor=EMRVendor.EPIC, base_url="https://emr.example.com")

        client = EMRIntegrationClient(config)

        record = client.get_patient_record("PAT001")

        # Should return None since not implemented
        assert record is None

    def test_validate_patient_identity_no_record(self):
        """Test patient identity validation with no record."""
        config = EMRConfig(vendor=EMRVendor.CERNER, base_url="https://emr.example.com")

        client = EMRIntegrationClient(config)

        valid = client.validate_patient_identity(
            patient_id="PAT001", mrn="MRN12345", date_of_birth=datetime(1980, 1, 1)
        )

        # Should return False since record not found
        assert valid is False

    def test_get_clinical_notes_not_implemented(self):
        """Test clinical notes retrieval (not implemented)."""
        config = EMRConfig(vendor=EMRVendor.ALLSCRIPTS, base_url="https://emr.example.com")

        client = EMRIntegrationClient(config)

        notes = client.get_clinical_notes("PAT001")

        # Should return empty list since not implemented
        assert len(notes) == 0

    def test_get_clinical_notes_with_type_filter(self):
        """Test clinical notes retrieval with type filter."""
        config = EMRConfig(vendor=EMRVendor.MEDITECH, base_url="https://emr.example.com")

        client = EMRIntegrationClient(config)

        notes = client.get_clinical_notes("PAT001", note_type="progress")

        assert len(notes) == 0

    def test_deliver_result_to_emr_not_implemented(self):
        """Test result delivery to EMR (not implemented)."""
        config = EMRConfig(vendor=EMRVendor.EPIC, base_url="https://emr.example.com")

        client = EMRIntegrationClient(config)

        result_data = {
            "prediction": "Positive",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
        }

        success = client.deliver_result_to_emr("PAT001", result_data)

        # Should return False since not implemented
        assert success is False

    def test_get_statistics(self):
        """Test get statistics."""
        config = EMRConfig(vendor=EMRVendor.CERNER, base_url="https://emr.example.com", timeout=45)

        client = EMRIntegrationClient(config)

        stats = client.get_statistics()

        assert stats["vendor"] == "cerner"
        assert stats["base_url"] == "https://emr.example.com"
        assert stats["timeout"] == 45


class TestEMRIntegrationFactory:
    """Tests for EMRIntegrationFactory."""

    def test_create_epic_client(self):
        """Test create Epic client."""
        client = EMRIntegrationFactory.create_client(
            vendor="epic", base_url="https://epic.example.com", client_id="epic_client", timeout=60
        )

        assert client.vendor == EMRVendor.EPIC
        assert client.config.base_url == "https://epic.example.com"
        assert client.config.client_id == "epic_client"
        assert client.config.timeout == 60

    def test_create_cerner_client(self):
        """Test create Cerner client."""
        client = EMRIntegrationFactory.create_client(
            vendor="cerner", base_url="https://cerner.example.com", api_key="cerner_key"
        )

        assert client.vendor == EMRVendor.CERNER
        assert client.config.api_key == "cerner_key"

    def test_create_generic_client(self):
        """Test create generic client."""
        client = EMRIntegrationFactory.create_client(
            vendor="unknown", base_url="https://emr.example.com"
        )

        # Unknown vendor should default to GENERIC
        assert client.vendor == EMRVendor.GENERIC

    def test_create_client_with_all_params(self):
        """Test create client with all parameters."""
        client = EMRIntegrationFactory.create_client(
            vendor="allscripts",
            base_url="https://allscripts.example.com",
            client_id="test_client",
            client_secret="test_secret",
            api_key="test_key",
            timeout=90,
        )

        assert client.vendor == EMRVendor.ALLSCRIPTS
        assert client.config.client_id == "test_client"
        assert client.config.client_secret == "test_secret"
        assert client.config.api_key == "test_key"
        assert client.config.timeout == 90


class TestDataModels:
    """Tests for data models."""

    def test_patient_record_creation(self):
        """Test PatientRecord creation."""
        record = PatientRecord(
            patient_id="PAT001",
            mrn="MRN12345",
            first_name="John",
            last_name="Doe",
            date_of_birth=datetime(1980, 1, 1),
            sex="M",
            contact_info={"phone": "555-1234"},
            insurance_info={"provider": "Blue Cross"},
        )

        assert record.patient_id == "PAT001"
        assert record.mrn == "MRN12345"
        assert record.first_name == "John"
        assert record.contact_info["phone"] == "555-1234"

    def test_clinical_note_creation(self):
        """Test ClinicalNote creation."""
        note = ClinicalNote(
            note_id="NOTE001",
            patient_id="PAT001",
            note_type="progress",
            author="Dr. Smith",
            timestamp=datetime(2026, 4, 15, 10, 30),
            content="Patient doing well.",
            status="final",
        )

        assert note.note_id == "NOTE001"
        assert note.patient_id == "PAT001"
        assert note.note_type == "progress"
        assert note.status == "final"

    def test_emr_config_creation(self):
        """Test EMRConfig creation."""
        config = EMRConfig(
            vendor=EMRVendor.EPIC,
            base_url="https://epic.example.com",
            client_id="test_client",
            client_secret="test_secret",
            timeout=45,
        )

        assert config.vendor == EMRVendor.EPIC
        assert config.base_url == "https://epic.example.com"
        assert config.timeout == 45
