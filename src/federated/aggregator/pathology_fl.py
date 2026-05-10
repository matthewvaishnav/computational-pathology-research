"""Production adapter for PathologyFL aggregation."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.federated.common.data_models import ClientUpdate
from src.federated.pathology_fl import (
    CancerType,
    HospitalMetadata,
    HospitalType,
    PathologyFederatedAggregator,
    SlideQuality,
)

from .base import BaseAggregator


class PathologyFLAggregator(BaseAggregator):
    """Adapter that exposes PathologyFL through the standard aggregator API."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        default_cancer_type: str = "general",
    ):
        super().__init__()
        self.algorithm_name = "PathologyFL"
        self.default_cancer_type = default_cancer_type
        self.pathology_aggregator = PathologyFederatedAggregator(alpha=alpha, beta=beta)

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates with PathologyFL metadata and quality weights."""

        if not client_updates:
            raise ValueError("Cannot aggregate empty list of client updates")

        cancer_type = self._resolve_cancer_type(client_updates)
        gradients = {update.client_id: update.gradients for update in client_updates}
        hospital_metadata = {
            update.client_id: self._parse_hospital_metadata(update)
            for update in client_updates
        }
        slide_quality = {
            update.client_id: self._parse_slide_quality(update)
            for update in client_updates
        }

        return self.pathology_aggregator.aggregate_updates(
            gradients,
            hospital_metadata,
            slide_quality,
            cancer_type,
        )

    def _resolve_cancer_type(self, client_updates: List[ClientUpdate]) -> CancerType:
        cancer_types = {
            (update.cancer_type or self.default_cancer_type).lower()
            for update in client_updates
        }
        if len(cancer_types) != 1:
            raise ValueError("PathologyFL requires all client updates to share one cancer_type")
        return CancerType(cancer_types.pop())

    def _parse_hospital_metadata(self, update: ClientUpdate) -> HospitalMetadata:
        raw_metadata = update.hospital_metadata
        if isinstance(raw_metadata, HospitalMetadata):
            return raw_metadata
        if not raw_metadata:
            raise ValueError(f"PathologyFL requires hospital_metadata for {update.client_id}")

        hospital_type = self._parse_hospital_type(raw_metadata.get("hospital_type"))
        specialties = [
            self._parse_cancer_type(specialty)
            for specialty in raw_metadata.get("cancer_specialties", [])
        ]

        return HospitalMetadata(
            hospital_id=str(raw_metadata.get("hospital_id") or update.client_id),
            hospital_type=hospital_type,
            annual_cases=int(raw_metadata["annual_cases"]),
            cancer_specialties=specialties,
            diagnostic_accuracy=float(raw_metadata["diagnostic_accuracy"]),
            years_experience=int(raw_metadata["years_experience"]),
        )

    def _parse_slide_quality(self, update: ClientUpdate) -> SlideQuality:
        raw_quality = update.slide_quality
        if isinstance(raw_quality, SlideQuality):
            return raw_quality
        if not raw_quality:
            raise ValueError(f"PathologyFL requires slide_quality for {update.client_id}")

        return SlideQuality(
            image_sharpness=float(raw_quality["image_sharpness"]),
            stain_consistency=float(raw_quality["stain_consistency"]),
            label_confidence=float(raw_quality["label_confidence"]),
            artifact_level=float(raw_quality["artifact_level"]),
        )

    def _parse_hospital_type(self, value: Any) -> HospitalType:
        if isinstance(value, HospitalType):
            return value
        aliases = {"community": "community_hospital", "rural": "rural_hospital"}
        normalized = aliases.get(str(value).lower(), str(value).lower())
        return HospitalType(normalized)

    def _parse_cancer_type(self, value: Any) -> CancerType:
        if isinstance(value, CancerType):
            return value
        return CancerType(str(value).lower())
