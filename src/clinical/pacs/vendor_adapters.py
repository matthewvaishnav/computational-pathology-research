"""Multi-vendor PACS adapter implementations for HistoCore DICOM integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind

from .data_models import PACSEndpoint, PACSVendor

IMPLICIT_VR_LE = "1.2.840.10008.1.2"
EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
JPEG_LOSSLESS_P14 = "1.2.840.10008.1.2.4.57"
JPEG_LOSSLESS_SV1 = "1.2.840.10008.1.2.4.70"
JPEG_LS_LOSSLESS = "1.2.840.10008.1.2.4.80"
JPEG_LS_NEAR_LOSSLESS = "1.2.840.10008.1.2.4.81"
JPEG2000_LOSSLESS = "1.2.840.10008.1.2.4.90"
JPEG2000_LOSSY = "1.2.840.10008.1.2.4.91"


class VendorAdapter(ABC):
    """Abstract base class for vendor-specific PACS adapters."""

    vendor: PACSVendor

    @abstractmethod
    def get_preferred_transfer_syntaxes(self) -> List[str]:
        """Return transfer syntax UIDs in preference order (best first)."""
        ...

    @abstractmethod
    def normalize_tags(self, dataset: Dataset) -> Dataset:
        """Strip/remap vendor private tags to DICOM standard equivalents."""
        ...

    @abstractmethod
    def get_query_model(self) -> str:
        """Return preferred C-FIND SOP class UID."""
        ...

    @abstractmethod
    def apply_optimizations(self, ae: AE) -> None:
        """Apply vendor-specific AE settings."""
        ...

    @abstractmethod
    def get_private_tag_blocks(self) -> Dict[str, str]:
        """Return {private_creator: description} for known vendor private blocks."""
        ...

    def detect_from_dataset(self, dataset: Dataset) -> bool:
        """Detect if a dataset originated from this vendor via manufacturer attributes."""
        return False


def _strip_private_groups(
    dataset: Dataset,
    groups: set,
    creator_prefix: Optional[str] = None,
) -> None:
    """Remove private tags from specified groups, optionally filtering by creator prefix."""
    tags_to_remove = set()

    for tag in list(dataset.keys()):
        if not tag.is_private or tag.group not in groups:
            continue

        # Private creator tags have element in 0x0010..0x00FF
        if 0x0010 <= tag.element <= 0x00FF:
            creator = str(dataset[tag].value)
            if creator_prefix is None or creator.startswith(creator_prefix):
                tags_to_remove.add(tag)
                block_base = tag.element << 8
                for dtag in list(dataset.keys()):
                    if dtag.group == tag.group and block_base <= dtag.element <= block_base + 0xFF:
                        tags_to_remove.add(dtag)
        elif creator_prefix is None:
            # No creator filter: remove all private elements in specified groups
            tags_to_remove.add(tag)

    for tag in tags_to_remove:
        if tag in dataset:
            del dataset[tag]


class GEAdapter(VendorAdapter):
    """Vendor adapter for GE Healthcare PACS systems."""

    vendor = PACSVendor.GE

    def get_preferred_transfer_syntaxes(self) -> List[str]:
        return [JPEG2000_LOSSLESS, JPEG_LOSSLESS_SV1, EXPLICIT_VR_LE, IMPLICIT_VR_LE]

    def normalize_tags(self, dataset: Dataset) -> Dataset:
        # Map (0009,0010) private creator to Manufacturer if Manufacturer is absent
        from pydicom.tag import Tag

        creator_tag = Tag(0x0009, 0x0010)
        if creator_tag in dataset and not dataset.get("Manufacturer"):
            dataset.Manufacturer = str(dataset[creator_tag].value)

        _strip_private_groups(
            dataset,
            groups={0x0009, 0x0019, 0x0021, 0x0025, 0x0027},
            creator_prefix="GEMS_",
        )
        return dataset

    def get_query_model(self) -> str:
        return StudyRootQueryRetrieveInformationModelFind

    def apply_optimizations(self, ae: AE) -> None:
        ae.maximum_pdu_size = 65536

    def get_private_tag_blocks(self) -> Dict[str, str]:
        return {
            "GEMS_IDEN_01": "GE Identity",
            "GEMS_ACQU_01": "GE Acquisition",
            "GEMS_RELA_01": "GE Relations",
            "GEMS_STDY_01": "GE Study",
        }

    def detect_from_dataset(self, dataset: Dataset) -> bool:
        manufacturer = dataset.get("Manufacturer", "").upper()
        institution = dataset.get("InstitutionName", "")
        return (
            "GE" in manufacturer
            or "GEMS" in manufacturer
            or "GE Healthcare" in institution
        )


class PhilipsAdapter(VendorAdapter):
    """Vendor adapter for Philips IntelliSpace PACS systems."""

    vendor = PACSVendor.PHILIPS

    def get_preferred_transfer_syntaxes(self) -> List[str]:
        return [JPEG_LOSSLESS_SV1, JPEG_LS_LOSSLESS, EXPLICIT_VR_LE, IMPLICIT_VR_LE]

    def normalize_tags(self, dataset: Dataset) -> Dataset:
        from pydicom.tag import Tag

        # Map Philips (2001,1003) "Number of slices MR" to NumberOfFrames if present
        philips_slices_tag = Tag(0x2001, 0x1003)
        if philips_slices_tag in dataset and not dataset.get("NumberOfFrames"):
            dataset.NumberOfFrames = dataset[philips_slices_tag].value

        _strip_private_groups(
            dataset,
            groups={0x2001, 0x2005, 0x200D},
        )
        return dataset

    def get_query_model(self) -> str:
        return StudyRootQueryRetrieveInformationModelFind

    def apply_optimizations(self, ae: AE) -> None:
        ae.maximum_pdu_size = 32768

    def get_private_tag_blocks(self) -> Dict[str, str]:
        return {
            "Philips Imaging DD 001": "Philips Imaging",
            "Philips MR Imaging DD 001": "Philips MR",
            "PHILIPS IMAGING": "Philips Imaging Legacy",
        }

    def detect_from_dataset(self, dataset: Dataset) -> bool:
        manufacturer = dataset.get("Manufacturer", "")
        return "PHILIPS" in manufacturer.upper() or "Philips" in manufacturer


class SiemensAdapter(VendorAdapter):
    """Vendor adapter for Siemens syngo PACS systems."""

    vendor = PACSVendor.SIEMENS

    def get_preferred_transfer_syntaxes(self) -> List[str]:
        return [JPEG_LS_LOSSLESS, JPEG2000_LOSSLESS, EXPLICIT_VR_LE, IMPLICIT_VR_LE]

    def normalize_tags(self, dataset: Dataset) -> Dataset:
        from pydicom.tag import Tag

        # CSA header data — strip without parsing (binary blobs not safe to remap inline)
        for csa_tag in [Tag(0x0029, 0x1010), Tag(0x0029, 0x1020)]:
            if csa_tag in dataset:
                del dataset[csa_tag]

        _strip_private_groups(
            dataset,
            groups={0x0019, 0x0021, 0x0029},
            creator_prefix="SIEMENS",
        )
        return dataset

    def get_query_model(self) -> str:
        return StudyRootQueryRetrieveInformationModelFind

    def apply_optimizations(self, ae: AE) -> None:
        ae.maximum_pdu_size = 65536

    def get_private_tag_blocks(self) -> Dict[str, str]:
        return {
            "SIEMENS CSA HEADER": "Siemens CSA Non-Image",
            "SIEMENS MED MG CS": "Siemens MG",
            "SIEMENS CT VA0 COAD": "Siemens CT",
            "SIEMENS CSA NON-IMAGE": "Siemens CSA",
        }

    def detect_from_dataset(self, dataset: Dataset) -> bool:
        return "SIEMENS" in dataset.get("Manufacturer", "").upper()


class AgfaAdapter(VendorAdapter):
    """Vendor adapter for Agfa Enterprise Imaging PACS systems."""

    vendor = PACSVendor.AGFA

    def get_preferred_transfer_syntaxes(self) -> List[str]:
        return [EXPLICIT_VR_LE, JPEG_LOSSLESS_SV1, IMPLICIT_VR_LE]

    def normalize_tags(self, dataset: Dataset) -> Dataset:
        _strip_private_groups(
            dataset,
            groups={0x0019},
            creator_prefix="AGFA",
        )
        return dataset

    def get_query_model(self) -> str:
        return StudyRootQueryRetrieveInformationModelFind

    def apply_optimizations(self, ae: AE) -> None:
        ae.maximum_pdu_size = 16384

    def get_private_tag_blocks(self) -> Dict[str, str]:
        return {
            "AGFA": "Agfa Private",
            "AGFA_ADC_Compact": "Agfa ADC",
        }

    def detect_from_dataset(self, dataset: Dataset) -> bool:
        return "AGFA" in dataset.get("Manufacturer", "").upper()


class GenericAdapter(VendorAdapter):
    """Fallback adapter for unknown or generic PACS systems."""

    vendor = PACSVendor.GENERIC

    def get_preferred_transfer_syntaxes(self) -> List[str]:
        return [EXPLICIT_VR_LE, IMPLICIT_VR_LE]

    def normalize_tags(self, dataset: Dataset) -> Dataset:
        return dataset

    def get_query_model(self) -> str:
        return StudyRootQueryRetrieveInformationModelFind

    def apply_optimizations(self, ae: AE) -> None:
        pass

    def get_private_tag_blocks(self) -> Dict[str, str]:
        return {}

    def detect_from_dataset(self, dataset: Dataset) -> bool:
        return False


_CONCRETE_ADAPTERS: List[type] = [GEAdapter, PhilipsAdapter, SiemensAdapter, AgfaAdapter]


class ConformanceNegotiator:
    """Negotiates DICOM transfer syntax conformance between client and server."""

    def negotiate_transfer_syntaxes(
        self,
        vendor_adapter: VendorAdapter,
        requested_abstract_syntax: str,
        remote_supported_syntaxes: Optional[List[str]] = None,
    ) -> List[str]:
        """Return agreed transfer syntaxes in vendor preference order.

        When remote capabilities are known, the result is the intersection
        of vendor preference and remote support — still in vendor order so
        the best common option is tried first.
        """
        preferred = vendor_adapter.get_preferred_transfer_syntaxes()

        if remote_supported_syntaxes is None:
            return preferred

        remote_set = set(remote_supported_syntaxes)
        return [ts for ts in preferred if ts in remote_set]

    def build_presentation_contexts(
        self,
        vendor_adapter: VendorAdapter,
        abstract_syntaxes: List[str],
    ) -> List[Dict[str, Any]]:
        """Build presentation context descriptors for AE configuration."""
        transfer_syntaxes = vendor_adapter.get_preferred_transfer_syntaxes()
        return [
            {"abstract_syntax": abs_syntax, "transfer_syntaxes": transfer_syntaxes}
            for abs_syntax in abstract_syntaxes
        ]


class VendorAdapterFactory:
    """Factory that creates and caches singleton vendor adapter instances."""

    _adapters: Dict[PACSVendor, VendorAdapter] = {}

    @classmethod
    def create(cls, vendor: PACSVendor) -> VendorAdapter:
        """Return singleton adapter for vendor; instantiate on first call."""
        if vendor not in cls._adapters:
            adapter_map: Dict[PACSVendor, type] = {
                PACSVendor.GE: GEAdapter,
                PACSVendor.PHILIPS: PhilipsAdapter,
                PACSVendor.SIEMENS: SiemensAdapter,
                PACSVendor.AGFA: AgfaAdapter,
                PACSVendor.GENERIC: GenericAdapter,
            }
            adapter_cls = adapter_map.get(vendor, GenericAdapter)
            cls._adapters[vendor] = adapter_cls()
        return cls._adapters[vendor]

    @classmethod
    def detect_from_dataset(cls, dataset: Dataset) -> VendorAdapter:
        """Detect vendor from dataset attributes; falls back to GenericAdapter."""
        for adapter_cls in _CONCRETE_ADAPTERS:
            adapter = cls.create(adapter_cls.vendor)
            if adapter.detect_from_dataset(dataset):
                return adapter
        return cls.create(PACSVendor.GENERIC)

    @classmethod
    def detect_from_endpoint(cls, endpoint: PACSEndpoint) -> VendorAdapter:
        """Return adapter matching the endpoint's declared vendor."""
        return cls.create(endpoint.vendor)
