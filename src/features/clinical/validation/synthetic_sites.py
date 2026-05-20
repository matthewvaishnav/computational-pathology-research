"""
Synthetic hospital site generation for multi-site validation.

Creates realistic synthetic hospital environments with varying patient
populations, data quality, equipment, and operational characteristics
for comprehensive validation of medical AI systems.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HospitalType(Enum):
    """Types of hospital facilities."""

    ACADEMIC_MEDICAL_CENTER = "academic_medical_center"
    COMMUNITY_HOSPITAL = "community_hospital"
    REGIONAL_MEDICAL_CENTER = "regional_medical_center"
    SPECIALTY_CANCER_CENTER = "specialty_cancer_center"
    RURAL_HOSPITAL = "rural_hospital"


class EquipmentTier(Enum):
    """Equipment quality tiers."""

    PREMIUM = "premium"
    STANDARD = "standard"
    BASIC = "basic"


@dataclass
class PatientDemographics:
    """Patient population demographics."""

    site_id: str

    # Age distribution
    mean_age: float
    age_std: float
    pediatric_percentage: float  # Under 18
    geriatric_percentage: float  # Over 65

    # Gender distribution
    female_percentage: float

    # Ethnicity distribution
    caucasian_percentage: float
    african_american_percentage: float
    hispanic_percentage: float
    asian_percentage: float
    other_percentage: float

    # Socioeconomic factors
    insurance_coverage_percentage: float
    median_income: float

    # Disease prevalence (per 100,000)
    breast_cancer_prevalence: float
    lung_cancer_prevalence: float
    prostate_cancer_prevalence: float
    colon_cancer_prevalence: float
    melanoma_prevalence: float


@dataclass
class DataQualityProfile:
    """Data quality characteristics for a site."""

    site_id: str

    # Image quality metrics
    mean_image_resolution: Tuple[int, int]  # (width, height)
    image_quality_score: float  # 0-1, higher is better
    compression_artifacts_rate: float  # 0-1, lower is better
    staining_consistency_score: float  # 0-1, higher is better

    # Data completeness
    complete_clinical_data_rate: float  # 0-1
    missing_demographics_rate: float  # 0-1
    missing_history_rate: float  # 0-1

    # Annotation quality
    expert_annotation_rate: float  # 0-1, rate of expert vs resident annotations
    inter_annotator_agreement: float  # 0-1, Cohen's kappa
    annotation_detail_level: float  # 0-1, detail of annotations

    # Technical factors
    scanner_calibration_score: float  # 0-1, higher is better
    color_consistency_score: float  # 0-1, higher is better
    focus_quality_score: float  # 0-1, higher is better


@dataclass
class OperationalProfile:
    """Operational characteristics of a site."""

    site_id: str

    # Staffing
    pathologists_count: int
    residents_count: int
    technicians_count: int

    # Workload
    cases_per_day: int
    turnaround_time_hours: float
    weekend_operations: bool

    # Technology adoption
    digital_pathology_adoption: float  # 0-1
    ai_familiarity_score: float  # 0-1
    it_infrastructure_score: float  # 0-1

    # Quality metrics
    error_rate: float  # 0-1
    second_opinion_rate: float  # 0-1
    external_consultation_rate: float  # 0-1


@dataclass
class SyntheticSite:
    """Complete synthetic hospital site."""

    site_id: str
    site_name: str
    hospital_type: HospitalType
    location: Dict[str, str]  # city, state, country

    # Site characteristics
    bed_count: int
    annual_case_volume: int
    equipment_tier: EquipmentTier

    # Detailed profiles
    demographics: PatientDemographics
    data_quality: DataQualityProfile
    operations: OperationalProfile

    # Validation parameters
    validation_case_count: int
    ground_truth_availability: float  # 0-1

    # Metadata
    creation_timestamp: float
    description: str


class SyntheticSiteGenerator:
    """
    Generates realistic synthetic hospital sites for validation.

    Creates diverse hospital environments with varying characteristics
    to test model performance across different real-world conditions.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize synthetic site generator."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Site templates for different hospital types
        self.site_templates = {
            HospitalType.ACADEMIC_MEDICAL_CENTER: {
                "bed_count_range": (400, 800),
                "case_volume_range": (15000, 30000),
                "equipment_tier": EquipmentTier.PREMIUM,
                "pathologists_range": (8, 15),
                "data_quality_base": 0.9,
                "ai_familiarity_base": 0.8,
            },
            HospitalType.COMMUNITY_HOSPITAL: {
                "bed_count_range": (100, 300),
                "case_volume_range": (3000, 8000),
                "equipment_tier": EquipmentTier.STANDARD,
                "pathologists_range": (2, 6),
                "data_quality_base": 0.75,
                "ai_familiarity_base": 0.5,
            },
            HospitalType.REGIONAL_MEDICAL_CENTER: {
                "bed_count_range": (200, 500),
                "case_volume_range": (8000, 18000),
                "equipment_tier": EquipmentTier.STANDARD,
                "pathologists_range": (4, 10),
                "data_quality_base": 0.8,
                "ai_familiarity_base": 0.6,
            },
            HospitalType.SPECIALTY_CANCER_CENTER: {
                "bed_count_range": (50, 200),
                "case_volume_range": (5000, 12000),
                "equipment_tier": EquipmentTier.PREMIUM,
                "pathologists_range": (3, 8),
                "data_quality_base": 0.95,
                "ai_familiarity_base": 0.9,
            },
            HospitalType.RURAL_HOSPITAL: {
                "bed_count_range": (25, 100),
                "case_volume_range": (500, 2000),
                "equipment_tier": EquipmentTier.BASIC,
                "pathologists_range": (1, 3),
                "data_quality_base": 0.65,
                "ai_familiarity_base": 0.3,
            },
        }

        # Geographic regions with different demographics
        self.geographic_regions = {
            "northeast": {
                "locations": [
                    {"city": "Boston", "state": "MA"},
                    {"city": "New York", "state": "NY"},
                    {"city": "Philadelphia", "state": "PA"},
                ],
                "demographics": {
                    "caucasian_base": 0.65,
                    "african_american_base": 0.15,
                    "hispanic_base": 0.12,
                    "asian_base": 0.06,
                    "median_income_base": 65000,
                },
            },
            "southeast": {
                "locations": [
                    {"city": "Atlanta", "state": "GA"},
                    {"city": "Miami", "state": "FL"},
                    {"city": "Charlotte", "state": "NC"},
                ],
                "demographics": {
                    "caucasian_base": 0.55,
                    "african_american_base": 0.25,
                    "hispanic_base": 0.15,
                    "asian_base": 0.04,
                    "median_income_base": 52000,
                },
            },
            "midwest": {
                "locations": [
                    {"city": "Chicago", "state": "IL"},
                    {"city": "Detroit", "state": "MI"},
                    {"city": "Cleveland", "state": "OH"},
                ],
                "demographics": {
                    "caucasian_base": 0.70,
                    "african_american_base": 0.18,
                    "hispanic_base": 0.08,
                    "asian_base": 0.03,
                    "median_income_base": 58000,
                },
            },
            "west": {
                "locations": [
                    {"city": "Los Angeles", "state": "CA"},
                    {"city": "Seattle", "state": "WA"},
                    {"city": "Denver", "state": "CO"},
                ],
                "demographics": {
                    "caucasian_base": 0.50,
                    "african_american_base": 0.08,
                    "hispanic_base": 0.30,
                    "asian_base": 0.10,
                    "median_income_base": 72000,
                },
            },
            "southwest": {
                "locations": [
                    {"city": "Phoenix", "state": "AZ"},
                    {"city": "Dallas", "state": "TX"},
                    {"city": "Houston", "state": "TX"},
                ],
                "demographics": {
                    "caucasian_base": 0.45,
                    "african_american_base": 0.12,
                    "hispanic_base": 0.35,
                    "asian_base": 0.06,
                    "median_income_base": 55000,
                },
            },
        }

        logger.info("Initialized synthetic site generator")

    def generate_sites(self, count: int = 5) -> List[SyntheticSite]:
        """
        Generate multiple synthetic hospital sites.

        Args:
            count: Number of sites to generate

        Returns:
            List of synthetic sites with diverse characteristics
        """
        sites = []

        # Ensure diversity in hospital types
        hospital_types = list(HospitalType)
        selected_types = []

        for i in range(count):
            # Select hospital type (ensure diversity)
            if i < len(hospital_types):
                hospital_type = hospital_types[i]
            else:
                hospital_type = random.choice(hospital_types)

            selected_types.append(hospital_type)

            # Generate site
            site = self._generate_single_site(f"site_{i+1:02d}", hospital_type)
            sites.append(site)

        logger.info(f"Generated {count} synthetic sites: {[s.hospital_type.value for s in sites]}")

        return sites

    def _generate_single_site(self, site_id: str, hospital_type: HospitalType) -> SyntheticSite:
        """Generate a single synthetic site."""
        template = self.site_templates[hospital_type]

        # Select geographic region and location
        region_name = random.choice(list(self.geographic_regions.keys()))
        region = self.geographic_regions[region_name]
        location = random.choice(region["locations"])
        location["country"] = "USA"

        # Generate basic site characteristics
        bed_count = random.randint(*template["bed_count_range"])
        annual_case_volume = random.randint(*template["case_volume_range"])

        # Generate site name
        site_name = self._generate_site_name(location, hospital_type)

        # Generate demographics
        demographics = self._generate_demographics(site_id, region)

        # Generate data quality profile
        data_quality = self._generate_data_quality(site_id, template)

        # Generate operational profile
        operations = self._generate_operations(site_id, template, annual_case_volume)

        # Calculate validation parameters
        validation_case_count = min(1000, annual_case_volume // 10)  # 10% of cases, max 1000
        ground_truth_availability = 0.8 + np.random.normal(0, 0.1)  # 80% ± 10%
        ground_truth_availability = np.clip(ground_truth_availability, 0.5, 1.0)

        return SyntheticSite(
            site_id=site_id,
            site_name=site_name,
            hospital_type=hospital_type,
            location=location,
            bed_count=bed_count,
            annual_case_volume=annual_case_volume,
            equipment_tier=template["equipment_tier"],
            demographics=demographics,
            data_quality=data_quality,
            operations=operations,
            validation_case_count=validation_case_count,
            ground_truth_availability=ground_truth_availability,
            creation_timestamp=datetime.now().timestamp(),
            description=f"{hospital_type.value.replace('_', ' ').title()} in {location['city']}, {location['state']}",
        )

    def _generate_site_name(self, location: Dict[str, str], hospital_type: HospitalType) -> str:
        """Generate realistic hospital name."""
        city = location["city"]

        if hospital_type == HospitalType.ACADEMIC_MEDICAL_CENTER:
            prefixes = ["University of", f"{city} University", f"{city} Medical College"]
            suffixes = ["Medical Center", "Hospital", "Health System"]
        elif hospital_type == HospitalType.SPECIALTY_CANCER_CENTER:
            prefixes = [f"{city}", f"{city} Regional", "Comprehensive"]
            suffixes = ["Cancer Center", "Oncology Institute", "Cancer Hospital"]
        elif hospital_type == HospitalType.RURAL_HOSPITAL:
            prefixes = [f"{city}", f"{city} Community", f"{city} General"]
            suffixes = ["Hospital", "Medical Center", "Healthcare"]
        else:
            prefixes = [f"{city}", f"{city} Regional", f"{city} General", "St. Mary's", "Memorial"]
            suffixes = ["Hospital", "Medical Center", "Health System", "Healthcare"]

        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)

        return f"{prefix} {suffix}"

    def _generate_demographics(self, site_id: str, region: Dict) -> PatientDemographics:
        """Generate patient demographics for site."""
        demo_base = region["demographics"]

        # Add variation to base demographics
        variation = 0.1  # 10% variation

        caucasian_pct = demo_base["caucasian_base"] + np.random.normal(0, variation)
        african_american_pct = demo_base["african_american_base"] + np.random.normal(0, variation)
        hispanic_pct = demo_base["hispanic_base"] + np.random.normal(0, variation)
        asian_pct = demo_base["asian_base"] + np.random.normal(0, variation)

        # Normalize to sum to 1
        total = caucasian_pct + african_american_pct + hispanic_pct + asian_pct
        other_pct = max(0, 1 - total)

        # Ensure all percentages are positive
        caucasian_pct = max(0.1, caucasian_pct)
        african_american_pct = max(0.05, african_american_pct)
        hispanic_pct = max(0.05, hispanic_pct)
        asian_pct = max(0.02, asian_pct)

        # Renormalize
        total = caucasian_pct + african_american_pct + hispanic_pct + asian_pct + other_pct
        caucasian_pct /= total
        african_american_pct /= total
        hispanic_pct /= total
        asian_pct /= total
        other_pct /= total

        # Generate disease prevalence (per 100,000)
        # Base rates with regional variation
        breast_cancer_base = 125  # per 100,000 women
        lung_cancer_base = 60
        prostate_cancer_base = 110  # per 100,000 men
        colon_cancer_base = 40
        melanoma_base = 25

        return PatientDemographics(
            site_id=site_id,
            mean_age=65 + np.random.normal(0, 5),
            age_std=18 + np.random.normal(0, 2),
            pediatric_percentage=0.05 + np.random.normal(0, 0.02),
            geriatric_percentage=0.35 + np.random.normal(0, 0.05),
            female_percentage=0.52 + np.random.normal(0, 0.03),
            caucasian_percentage=caucasian_pct,
            african_american_percentage=african_american_pct,
            hispanic_percentage=hispanic_pct,
            asian_percentage=asian_pct,
            other_percentage=other_pct,
            insurance_coverage_percentage=0.85 + np.random.normal(0, 0.1),
            median_income=demo_base["median_income_base"] + np.random.normal(0, 10000),
            breast_cancer_prevalence=breast_cancer_base + np.random.normal(0, 15),
            lung_cancer_prevalence=lung_cancer_base + np.random.normal(0, 10),
            prostate_cancer_prevalence=prostate_cancer_base + np.random.normal(0, 15),
            colon_cancer_prevalence=colon_cancer_base + np.random.normal(0, 8),
            melanoma_prevalence=melanoma_base + np.random.normal(0, 5),
        )

    def _generate_data_quality(self, site_id: str, template: Dict) -> DataQualityProfile:
        """Generate data quality profile for site."""
        base_quality = template["data_quality_base"]

        # Equipment tier affects image quality
        equipment_multiplier = {
            EquipmentTier.PREMIUM: 1.0,
            EquipmentTier.STANDARD: 0.9,
            EquipmentTier.BASIC: 0.8,
        }[template["equipment_tier"]]

        image_quality = base_quality * equipment_multiplier + np.random.normal(0, 0.05)
        image_quality = np.clip(image_quality, 0.5, 1.0)

        return DataQualityProfile(
            site_id=site_id,
            mean_image_resolution=(
                int(2048 + np.random.normal(0, 200)),
                int(2048 + np.random.normal(0, 200)),
            ),
            image_quality_score=image_quality,
            compression_artifacts_rate=max(
                0, 0.1 - image_quality * 0.1 + np.random.normal(0, 0.02)
            ),
            staining_consistency_score=base_quality + np.random.normal(0, 0.1),
            complete_clinical_data_rate=base_quality + np.random.normal(0, 0.1),
            missing_demographics_rate=max(0, 0.2 - base_quality * 0.2 + np.random.normal(0, 0.05)),
            missing_history_rate=max(0, 0.15 - base_quality * 0.15 + np.random.normal(0, 0.05)),
            expert_annotation_rate=base_quality + np.random.normal(0, 0.1),
            inter_annotator_agreement=0.7 + base_quality * 0.2 + np.random.normal(0, 0.05),
            annotation_detail_level=base_quality + np.random.normal(0, 0.1),
            scanner_calibration_score=image_quality + np.random.normal(0, 0.05),
            color_consistency_score=image_quality + np.random.normal(0, 0.05),
            focus_quality_score=image_quality + np.random.normal(0, 0.05),
        )

    def _generate_operations(
        self, site_id: str, template: Dict, case_volume: int
    ) -> OperationalProfile:
        """Generate operational profile for site."""
        pathologists_count = random.randint(*template["pathologists_range"])

        # Scale other staff based on pathologists
        residents_count = max(0, pathologists_count - 2 + random.randint(-1, 3))
        technicians_count = pathologists_count * 2 + random.randint(-2, 4)

        # Calculate workload
        cases_per_day = case_volume // 250  # Assuming 250 working days
        turnaround_time = 24 + np.random.normal(0, 6)  # Base 24 hours ± 6

        return OperationalProfile(
            site_id=site_id,
            pathologists_count=pathologists_count,
            residents_count=residents_count,
            technicians_count=technicians_count,
            cases_per_day=cases_per_day,
            turnaround_time_hours=max(4, turnaround_time),
            weekend_operations=random.choice([True, False]),
            digital_pathology_adoption=template["data_quality_base"] + np.random.normal(0, 0.1),
            ai_familiarity_score=template["ai_familiarity_base"] + np.random.normal(0, 0.1),
            it_infrastructure_score=template["data_quality_base"] + np.random.normal(0, 0.1),
            error_rate=max(
                0.001, 0.02 - template["data_quality_base"] * 0.015 + np.random.normal(0, 0.005)
            ),
            second_opinion_rate=0.1 + np.random.normal(0, 0.03),
            external_consultation_rate=0.05 + np.random.normal(0, 0.02),
        )

    def export_sites(self, sites: List[SyntheticSite], filepath: str) -> None:
        """Export synthetic sites to JSON file."""
        sites_data = []

        for site in sites:
            site_dict = asdict(site)
            # Convert enums to strings
            site_dict["hospital_type"] = site.hospital_type.value
            site_dict["equipment_tier"] = site.equipment_tier.value
            sites_data.append(site_dict)

        with open(filepath, "w") as f:
            json.dump(sites_data, f, indent=2, default=str)

        logger.info(f"Exported {len(sites)} synthetic sites to {filepath}")

    def generate_validation_dataset(
        self, site: SyntheticSite, disease_types: List[str] = None
    ) -> Dict[str, Any]:
        """Generate synthetic validation dataset for a site."""
        if disease_types is None:
            disease_types = ["breast", "lung", "prostate", "colon", "melanoma"]

        dataset = {
            "site_id": site.site_id,
            "site_name": site.site_name,
            "total_cases": site.validation_case_count,
            "ground_truth_availability": site.ground_truth_availability,
            "disease_distribution": {},
            "data_characteristics": {
                "image_quality_score": site.data_quality.image_quality_score,
                "staining_consistency": site.data_quality.staining_consistency_score,
                "annotation_quality": site.data_quality.expert_annotation_rate,
            },
        }

        # Generate disease distribution based on site demographics
        total_cases = site.validation_case_count
        remaining_cases = total_cases

        for disease in disease_types:
            # Get prevalence rate
            prevalence_attr = f"{disease}_cancer_prevalence"
            if hasattr(site.demographics, prevalence_attr):
                prevalence = getattr(site.demographics, prevalence_attr)
                # Convert prevalence per 100,000 to proportion
                base_proportion = prevalence / 100000 * 10  # Scale up for validation
            else:
                base_proportion = 0.1  # Default 10%

            # Add variation
            proportion = base_proportion + np.random.normal(0, base_proportion * 0.2)
            proportion = max(0.05, min(0.4, proportion))  # Clamp between 5% and 40%

            if disease == disease_types[-1]:  # Last disease gets remaining cases
                case_count = remaining_cases
            else:
                case_count = int(total_cases * proportion)
                remaining_cases -= case_count

            dataset["disease_distribution"][disease] = {
                "case_count": case_count,
                "proportion": case_count / total_cases,
                "ground_truth_available": int(case_count * site.ground_truth_availability),
            }

        return dataset


if __name__ == "__main__":
    # Demo: Synthetic site generation

    print("=== Synthetic Hospital Sites Generation Demo ===\n")

    # Create generator
    generator = SyntheticSiteGenerator(random_seed=42)

    # Generate 5 diverse sites
    sites = generator.generate_sites(count=5)

    print(f"Generated {len(sites)} synthetic hospital sites:\n")

    for i, site in enumerate(sites, 1):
        print(f"{i}. {site.site_name}")
        print(f"   Type: {site.hospital_type.value.replace('_', ' ').title()}")
        print(f"   Location: {site.location['city']}, {site.location['state']}")
        print(f"   Beds: {site.bed_count}, Annual Cases: {site.annual_case_volume:,}")
        print(f"   Equipment: {site.equipment_tier.value.title()}")
        print(f"   Data Quality: {site.data_quality.image_quality_score:.2f}")
        print(f"   Pathologists: {site.operations.pathologists_count}")
        print(f"   Validation Cases: {site.validation_case_count}")

        # Show demographics summary
        demo = site.demographics
        print(
            f"   Demographics: {demo.caucasian_percentage:.1%} Caucasian, "
            f"{demo.african_american_percentage:.1%} African American, "
            f"{demo.hispanic_percentage:.1%} Hispanic"
        )
        print()

    # Generate validation datasets
    print("--- Validation Datasets ---")
    for site in sites[:3]:  # Show first 3
        dataset = generator.generate_validation_dataset(site)
        print(f"\n{site.site_name}:")
        print(f"  Total cases: {dataset['total_cases']}")
        print(f"  Ground truth: {dataset['ground_truth_availability']:.1%}")
        print(f"  Disease distribution:")
        for disease, info in dataset["disease_distribution"].items():
            print(f"    {disease.title()}: {info['case_count']} cases ({info['proportion']:.1%})")

    # Export sites
    generator.export_sites(sites, "synthetic_sites.json")
    print(f"\nSynthetic sites exported to synthetic_sites.json")

    print("\n=== Demo Complete ===")
