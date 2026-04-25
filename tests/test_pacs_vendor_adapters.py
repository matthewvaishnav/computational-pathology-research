"""Property-based tests for PACS Multi-Vendor Support.

Feature: pacs-integration-system
Property 12: DICOM Conformance Negotiation
Property 13: Vendor Tag Normalization
Property 14: Vendor-Specific Optimization Selection
"""

from typing import List

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydicom.dataset import Dataset
from pydicom.dataelem import DataElement
from pydicom.tag import Tag
from pynetdicom import AE

from src.clinical.pacs.data_models import (
    PACSEndpoint,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
)
from src.clinical.pacs.vendor_adapters import (
    AgfaAdapter,
    ConformanceNegotiator,
    GEAdapter,
    GenericAdapter,
    PhilipsAdapter,
    SiemensAdapter,
    VendorAdapterFactory,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_endpoint(vendor: PACSVendor) -> PACSEndpoint:
    """Create a test PACS endpoint for a given vendor."""
    return PACSEndpoint(
        endpoint_id=f"test-{vendor.value}",
        ae_title=f"{vendor.value.upper()}_AE",
        host=f"pacs.{vendor.value}.local",
        port=11112,
        vendor=vendor,
        security_config=SecurityConfig(
            tls_enabled=False,
            verify_certificates=False,
            mutual_authentication=False,
        ),
        performance_config=PerformanceConfig(),
    )


# ---------------------------------------------------------------------------
# Property 12 — DICOM Conformance Negotiation
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 12: DICOM Conformance Negotiation
# For any PACS connection attempt, the highest common DICOM conformance level
# SHALL be negotiated between the client and server capabilities.


@given(vendor=st.sampled_from(list(PACSVendor)))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_12_conformance_negotiation_returns_vendor_preferences(vendor):
    """Conformance negotiation must return vendor-preferred transfer syntaxes."""
    adapter = VendorAdapterFactory.create(vendor)
    negotiator = ConformanceNegotiator()

    # Get vendor preferences
    preferred_syntaxes = adapter.get_preferred_transfer_syntaxes()

    # Negotiate without remote constraints (should return all preferences)
    negotiated = negotiator.negotiate_transfer_syntaxes(
        vendor_adapter=adapter,
        requested_abstract_syntax="1.2.840.10008.5.1.4.1.1.77.1.6",  # WSI Storage
        remote_supported_syntaxes=None,
    )

    # Must return vendor preferences in order
    assert negotiated == preferred_syntaxes
    assert len(negotiated) > 0


@given(
    vendor=st.sampled_from(list(PACSVendor)),
    remote_syntaxes=st.lists(
        st.sampled_from(
            [
                "1.2.840.10008.1.2",  # Implicit VR LE
                "1.2.840.10008.1.2.1",  # Explicit VR LE
                "1.2.840.10008.1.2.4.57",  # JPEG Lossless P14
                "1.2.840.10008.1.2.4.70",  # JPEG Lossless SV1
                "1.2.840.10008.1.2.4.80",  # JPEG-LS Lossless
                "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
            ]
        ),
        min_size=1,
        max_size=6,
        unique=True,
    ),
)
@settings(max_examples=100)
def test_property_12_conformance_negotiation_respects_remote_capabilities(vendor, remote_syntaxes):
    """Conformance negotiation must respect remote PACS capabilities."""
    adapter = VendorAdapterFactory.create(vendor)
    negotiator = ConformanceNegotiator()

    # Negotiate with remote constraints
    negotiated = negotiator.negotiate_transfer_syntaxes(
        vendor_adapter=adapter,
        requested_abstract_syntax="1.2.840.10008.5.1.4.1.1.77.1.6",
        remote_supported_syntaxes=remote_syntaxes,
    )

    # All negotiated syntaxes must be supported by remote
    remote_set = set(remote_syntaxes)
    assert all(ts in remote_set for ts in negotiated)

    # Negotiated syntaxes must be in vendor preference order
    preferred = adapter.get_preferred_transfer_syntaxes()
    preferred_indices = {ts: i for i, ts in enumerate(preferred)}

    for i in range(len(negotiated) - 1):
        current_idx = preferred_indices.get(negotiated[i], float("inf"))
        next_idx = preferred_indices.get(negotiated[i + 1], float("inf"))
        assert current_idx < next_idx, "Negotiated syntaxes not in vendor preference order"


@given(vendor=st.sampled_from(list(PACSVendor)))
@settings(max_examples=50)
def test_property_12_presentation_contexts_include_vendor_preferences(vendor):
    """Presentation contexts must include vendor-preferred transfer syntaxes."""
    adapter = VendorAdapterFactory.create(vendor)
    negotiator = ConformanceNegotiator()

    abstract_syntaxes = [
        "1.2.840.10008.5.1.4.1.1.77.1.6",  # WSI Storage
        "1.2.840.10008.5.1.4.1.1.77.1.1.1",  # VL Microscopic Image Storage
    ]

    contexts = negotiator.build_presentation_contexts(adapter, abstract_syntaxes)

    # Must have one context per abstract syntax
    assert len(contexts) == len(abstract_syntaxes)

    # Each context must include vendor preferences
    preferred = adapter.get_preferred_transfer_syntaxes()
    for context in contexts:
        assert context["abstract_syntax"] in abstract_syntaxes
        assert context["transfer_syntaxes"] == preferred


# ---------------------------------------------------------------------------
# Property 13 — Vendor Tag Normalization
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 13: Vendor Tag Normalization
# For any vendor-specific DICOM data, tag variations SHALL be handled
# transparently and normalized to standard representations.


@given(vendor=st.sampled_from([PACSVendor.GE, PACSVendor.PHILIPS, PACSVendor.SIEMENS, PACSVendor.AGFA]))
@settings(max_examples=50)
def test_property_13_vendor_tags_are_removed_after_normalization(vendor):
    """Vendor-specific private tags must be removed during normalization."""
    adapter = VendorAdapterFactory.create(vendor)

    # Create dataset with vendor-specific private tags
    ds = Dataset()
    ds.PatientID = "P001"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.Manufacturer = vendor.value.upper()

    # Add vendor-specific private tags based on vendor using DataElement
    if vendor == PACSVendor.GE:
        ds.add(DataElement(Tag(0x0009, 0x0010), "LO", "GEMS_IDEN_01"))
        ds.add(DataElement(Tag(0x0009, 0x1001), "LO", "GE Private Data"))
    elif vendor == PACSVendor.PHILIPS:
        ds.add(DataElement(Tag(0x2001, 0x0010), "LO", "Philips Imaging DD 001"))
        ds.add(DataElement(Tag(0x2001, 0x1003), "IS", "10"))  # Number of slices
    elif vendor == PACSVendor.SIEMENS:
        ds.add(DataElement(Tag(0x0029, 0x0010), "LO", "SIEMENS CSA HEADER"))
        ds.add(DataElement(Tag(0x0029, 0x1010), "OB", b"CSA binary data"))
    elif vendor == PACSVendor.AGFA:
        ds.add(DataElement(Tag(0x0019, 0x0010), "LO", "AGFA"))
        ds.add(DataElement(Tag(0x0019, 0x1001), "LO", "Agfa Private Data"))

    # Normalize
    normalized = adapter.normalize_tags(ds)

    # Vendor-specific private tags must be removed
    if vendor == PACSVendor.GE:
        assert Tag(0x0009, 0x1001) not in normalized
    elif vendor == PACSVendor.PHILIPS:
        # Philips (2001,1003) should be removed
        assert Tag(0x2001, 0x1003) not in normalized or "NumberOfFrames" in normalized
    elif vendor == PACSVendor.SIEMENS:
        assert Tag(0x0029, 0x1010) not in normalized
    elif vendor == PACSVendor.AGFA:
        assert Tag(0x0019, 0x1001) not in normalized

    # Standard tags must be preserved
    assert normalized.PatientID == "P001"
    assert normalized.StudyInstanceUID == "1.2.3.4.5"


@given(vendor=st.sampled_from([PACSVendor.GE, PACSVendor.PHILIPS]))
@settings(max_examples=30)
def test_property_13_vendor_tags_mapped_to_standard_equivalents(vendor):
    """Vendor-specific tags must be mapped to standard DICOM equivalents when possible."""
    adapter = VendorAdapterFactory.create(vendor)

    ds = Dataset()
    ds.PatientID = "P001"
    ds.StudyInstanceUID = "1.2.3.4.5"

    if vendor == PACSVendor.GE:
        # GE private creator tag can be mapped to Manufacturer
        ds.add(DataElement(Tag(0x0009, 0x0010), "LO", "GE Healthcare"))
        # Don't set Manufacturer initially
    elif vendor == PACSVendor.PHILIPS:
        # Philips Number of slices can be mapped to NumberOfFrames
        ds.add(DataElement(Tag(0x2001, 0x1003), "IS", "10"))
        # Don't set NumberOfFrames initially

    normalized = adapter.normalize_tags(ds)

    if vendor == PACSVendor.GE:
        # Manufacturer should be set from private tag
        assert "Manufacturer" in normalized
    elif vendor == PACSVendor.PHILIPS:
        # NumberOfFrames should be set from private tag
        assert "NumberOfFrames" in normalized
        assert int(normalized.NumberOfFrames) == 10


def test_property_13_generic_adapter_preserves_all_tags():
    """Generic adapter must preserve all tags without modification."""
    adapter = VendorAdapterFactory.create(PACSVendor.GENERIC)

    ds = Dataset()
    ds.PatientID = "P001"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.add(DataElement(Tag(0x0009, 0x0010), "LO", "Unknown Private Creator"))
    ds.add(DataElement(Tag(0x0009, 0x1001), "LO", "Unknown Private Data"))

    normalized = adapter.normalize_tags(ds)

    # All tags must be preserved
    assert normalized.PatientID == "P001"
    assert normalized.StudyInstanceUID == "1.2.3.4.5"
    assert Tag(0x0009, 0x0010) in normalized
    assert Tag(0x0009, 0x1001) in normalized


# ---------------------------------------------------------------------------
# Property 14 — Vendor-Specific Optimization Selection
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 14: Vendor-Specific Optimization Selection
# For any identified PACS vendor, appropriate vendor-specific optimizations
# SHALL be automatically applied based on the vendor identification.


@given(vendor=st.sampled_from(list(PACSVendor)))
@settings(max_examples=50)
def test_property_14_vendor_optimizations_applied_to_ae(vendor):
    """Vendor-specific optimizations must be applied to Application Entity."""
    adapter = VendorAdapterFactory.create(vendor)

    # Create AE
    ae = AE(ae_title="TEST_AE")
    initial_pdu_size = ae.maximum_pdu_size

    # Apply optimizations
    adapter.apply_optimizations(ae)

    # Verify optimizations were applied
    if vendor == PACSVendor.GE:
        assert ae.maximum_pdu_size == 65536
    elif vendor == PACSVendor.PHILIPS:
        assert ae.maximum_pdu_size == 32768
    elif vendor == PACSVendor.SIEMENS:
        assert ae.maximum_pdu_size == 65536
    elif vendor == PACSVendor.AGFA:
        assert ae.maximum_pdu_size == 16384
    elif vendor == PACSVendor.GENERIC:
        # Generic adapter should not modify PDU size
        assert ae.maximum_pdu_size == initial_pdu_size


@given(vendor=st.sampled_from(list(PACSVendor)))
@settings(max_examples=50)
def test_property_14_vendor_query_model_selection(vendor):
    """Vendor-specific query models must be selected correctly."""
    adapter = VendorAdapterFactory.create(vendor)

    query_model = adapter.get_query_model()

    # All vendors currently use StudyRootQueryRetrieveInformationModelFind
    # This test validates the interface exists and returns a valid SOP class
    assert query_model is not None
    assert isinstance(query_model, str) or hasattr(query_model, "__name__")


@given(vendor=st.sampled_from(list(PACSVendor)))
@settings(max_examples=50)
def test_property_14_vendor_transfer_syntax_preferences(vendor):
    """Vendor-specific transfer syntax preferences must be defined."""
    adapter = VendorAdapterFactory.create(vendor)

    transfer_syntaxes = adapter.get_preferred_transfer_syntaxes()

    # Must have at least one transfer syntax
    assert len(transfer_syntaxes) > 0

    # All transfer syntaxes must be valid UIDs
    for ts in transfer_syntaxes:
        assert isinstance(ts, str)
        assert len(ts) > 0
        # DICOM UIDs contain only digits and dots
        assert all(c.isdigit() or c == "." for c in ts)

    # Vendor-specific preferences
    if vendor == PACSVendor.GE:
        # GE prefers JPEG 2000
        assert "1.2.840.10008.1.2.4.90" in transfer_syntaxes  # JPEG2000 Lossless
    elif vendor == PACSVendor.PHILIPS:
        # Philips prefers JPEG Lossless
        assert "1.2.840.10008.1.2.4.70" in transfer_syntaxes  # JPEG Lossless SV1
    elif vendor == PACSVendor.SIEMENS:
        # Siemens prefers JPEG-LS
        assert "1.2.840.10008.1.2.4.80" in transfer_syntaxes  # JPEG-LS Lossless
    elif vendor == PACSVendor.AGFA:
        # Agfa prefers Explicit VR
        assert "1.2.840.10008.1.2.1" in transfer_syntaxes  # Explicit VR LE


# ---------------------------------------------------------------------------
# Vendor Detection Tests
# ---------------------------------------------------------------------------


def test_vendor_detection_from_dataset_ge():
    """GE vendor must be detected from dataset attributes."""
    ds = Dataset()
    ds.Manufacturer = "GE Healthcare"
    ds.PatientID = "P001"

    adapter = VendorAdapterFactory.detect_from_dataset(ds)

    assert adapter.vendor == PACSVendor.GE


def test_vendor_detection_from_dataset_philips():
    """Philips vendor must be detected from dataset attributes."""
    ds = Dataset()
    ds.Manufacturer = "PHILIPS"
    ds.PatientID = "P001"

    adapter = VendorAdapterFactory.detect_from_dataset(ds)

    assert adapter.vendor == PACSVendor.PHILIPS


def test_vendor_detection_from_dataset_siemens():
    """Siemens vendor must be detected from dataset attributes."""
    ds = Dataset()
    ds.Manufacturer = "SIEMENS"
    ds.PatientID = "P001"

    adapter = VendorAdapterFactory.detect_from_dataset(ds)

    assert adapter.vendor == PACSVendor.SIEMENS


def test_vendor_detection_from_dataset_agfa():
    """Agfa vendor must be detected from dataset attributes."""
    ds = Dataset()
    ds.Manufacturer = "AGFA"
    ds.PatientID = "P001"

    adapter = VendorAdapterFactory.detect_from_dataset(ds)

    assert adapter.vendor == PACSVendor.AGFA


def test_vendor_detection_from_dataset_unknown():
    """Unknown vendor must fall back to generic adapter."""
    ds = Dataset()
    ds.Manufacturer = "Unknown Vendor"
    ds.PatientID = "P001"

    adapter = VendorAdapterFactory.detect_from_dataset(ds)

    assert adapter.vendor == PACSVendor.GENERIC


def test_vendor_detection_from_endpoint():
    """Vendor must be detected from endpoint configuration."""
    endpoint = _make_endpoint(PACSVendor.GE)

    adapter = VendorAdapterFactory.detect_from_endpoint(endpoint)

    assert adapter.vendor == PACSVendor.GE


# ---------------------------------------------------------------------------
# Private Tag Block Tests
# ---------------------------------------------------------------------------


@given(vendor=st.sampled_from(list(PACSVendor)))
@settings(max_examples=50)
def test_vendor_private_tag_blocks_defined(vendor):
    """Vendor-specific private tag blocks must be defined."""
    adapter = VendorAdapterFactory.create(vendor)

    private_blocks = adapter.get_private_tag_blocks()

    # Must return a dictionary
    assert isinstance(private_blocks, dict)

    # Generic adapter has no private blocks
    if vendor == PACSVendor.GENERIC:
        assert len(private_blocks) == 0
    else:
        # Vendor adapters should have at least one private block
        assert len(private_blocks) > 0

        # All keys and values must be strings
        for creator, description in private_blocks.items():
            assert isinstance(creator, str)
            assert isinstance(description, str)
            assert len(creator) > 0
            assert len(description) > 0


# ---------------------------------------------------------------------------
# Factory Singleton Tests
# ---------------------------------------------------------------------------


def test_vendor_adapter_factory_returns_singletons():
    """Factory must return singleton instances for each vendor."""
    adapter1 = VendorAdapterFactory.create(PACSVendor.GE)
    adapter2 = VendorAdapterFactory.create(PACSVendor.GE)

    # Must be the same instance
    assert adapter1 is adapter2


def test_vendor_adapter_factory_creates_different_instances_per_vendor():
    """Factory must create different instances for different vendors."""
    ge_adapter = VendorAdapterFactory.create(PACSVendor.GE)
    philips_adapter = VendorAdapterFactory.create(PACSVendor.PHILIPS)

    # Must be different instances
    assert ge_adapter is not philips_adapter
    assert ge_adapter.vendor != philips_adapter.vendor
