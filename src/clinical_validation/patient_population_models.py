"""
Patient population modeling for clinical validation.

Models diverse patient populations with realistic disease patterns,
demographic distributions, comorbidities, and clinical presentations
for comprehensive AI system validation.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from datetime import datetime, timedelta
import random
from scipy import stats

logger = logging.getLogger(__name__)


class AgeGroup(Enum):
    """Patient age groups."""
    PEDIATRIC = "pediatric"      # 0-17
    YOUNG_ADULT = "young_adult"  # 18-39
    MIDDLE_AGED = "middle_aged"  # 40-64
    ELDERLY = "elderly"          # 65-79
    GERIATRIC = "geriatric"      # 80+


class SocioeconomicStatus(Enum):
    """Socioeconomic status categories."""
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"


@dataclass
class PatientProfile:
    """Individual patient profile."""
    patient_id: str
    site_id: str
    
    # Demographics
    age: int
    gender: str  # "M", "F"
    ethnicity: str
    
    # Socioeconomic
    insurance_type: str  # "private", "medicare", "medicaid", "uninsured"
    income_bracket: SocioeconomicStatus
    education_level: str  # "high_school", "college", "graduate"
    
    # Clinical history
    smoking_history: str  # "never", "former", "current"
    alcohol_use: str     # "none", "moderate", "heavy"
    family_history_cancer: bool
    
    # Comorbidities
    diabetes: bool
    hypertension: bool
    heart_disease: bool
    obesity: bool
    immunocompromised: bool
    
    # Disease-specific risk factors
    breast_cancer_risk_score: float  # 0-1
    lung_cancer_risk_score: float
    prostate_cancer_risk_score: float
    colon_cancer_risk_score: float
    melanoma_risk_score: float
    
    # Presentation characteristics
    symptom_duration_days: Optional[int]
    presentation_stage: Optional[str]  # "early", "intermediate", "advanced"
    diagnostic_delay_days: Optional[int]


@dataclass
class PopulationCharacteristics:
    """Population-level characteristics."""
    site_id: str
    population_name: str
    
    # Size and composition
    total_patients: int
    age_distribution: Dict[AgeGroup, float]
    gender_distribution: Dict[str, float]
    ethnicity_distribution: Dict[str, float]
    
    # Socioeconomic factors
    insurance_distribution: Dict[str, float]
    income_distribution: Dict[SocioeconomicStatus, float]
    education_distribution: Dict[str, float]
    
    # Health behaviors
    smoking_rates: Dict[str, float]
    alcohol_rates: Dict[str, float]
    
    # Disease prevalence (per 100,000)
    disease_prevalence: Dict[str, float]
    
    # Healthcare access
    screening_participation_rate: float
    diagnostic_delay_mean_days: float
    specialist_access_rate: float


class PatientPopulationGenerator:
    """
    Generates realistic patient populations for validation.
    
    Creates diverse patient cohorts with realistic demographic,
    socioeconomic, and clinical characteristics based on
    epidemiological data and regional variations.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize population generator."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Base disease prevalence rates (per 100,000)
        self.base_prevalence = {
            "breast_cancer": 125,    # per 100K women
            "lung_cancer": 60,       # per 100K
            "prostate_cancer": 110,  # per 100K men
            "colon_cancer": 40,      # per 100K
            "melanoma": 25          # per 100K
        }
        
        # Risk factor multipliers
        self.risk_multipliers = {
            "smoking": {
                "lung_cancer": 15.0,
                "bladder_cancer": 3.0,
                "colon_cancer": 1.5
            },
            "family_history": {
                "breast_cancer": 2.0,
                "prostate_cancer": 2.5,
                "colon_cancer": 2.0
            },
            "age_elderly": {
                "breast_cancer": 3.0,
                "lung_cancer": 4.0,
                "prostate_cancer": 8.0,
                "colon_cancer": 5.0
            },
            "obesity": {
                "breast_cancer": 1.3,
                "colon_cancer": 1.5,
                "endometrial_cancer": 2.0
            }
        }
        
        # Population templates by region/type
        self.population_templates = {
            "urban_diverse": {
                "ethnicity": {"caucasian": 0.45, "african_american": 0.25, "hispanic": 0.20, "asian": 0.08, "other": 0.02},
                "income": {"low": 0.25, "lower_middle": 0.20, "middle": 0.30, "upper_middle": 0.20, "high": 0.05},
                "education": {"high_school": 0.35, "college": 0.45, "graduate": 0.20},
                "smoking": {"never": 0.60, "former": 0.25, "current": 0.15},
                "screening_rate": 0.75
            },
            "suburban_affluent": {
                "ethnicity": {"caucasian": 0.70, "african_american": 0.10, "hispanic": 0.12, "asian": 0.06, "other": 0.02},
                "income": {"low": 0.10, "lower_middle": 0.15, "middle": 0.35, "upper_middle": 0.30, "high": 0.10},
                "education": {"high_school": 0.25, "college": 0.50, "graduate": 0.25},
                "smoking": {"never": 0.70, "former": 0.20, "current": 0.10},
                "screening_rate": 0.85
            },
            "rural_underserved": {
                "ethnicity": {"caucasian": 0.80, "african_american": 0.12, "hispanic": 0.06, "asian": 0.01, "other": 0.01},
                "income": {"low": 0.35, "lower_middle": 0.30, "middle": 0.25, "upper_middle": 0.08, "high": 0.02},
                "education": {"high_school": 0.55, "college": 0.35, "graduate": 0.10},
                "smoking": {"never": 0.45, "former": 0.30, "current": 0.25},
                "screening_rate": 0.55
            },
            "academic_medical": {
                "ethnicity": {"caucasian": 0.55, "african_american": 0.20, "hispanic": 0.15, "asian": 0.08, "other": 0.02},
                "income": {"low": 0.20, "lower_middle": 0.25, "middle": 0.30, "upper_middle": 0.20, "high": 0.05},
                "education": {"high_school": 0.30, "college": 0.45, "graduate": 0.25},
                "smoking": {"never": 0.65, "former": 0.25, "current": 0.10},
                "screening_rate": 0.80
            },
            "safety_net": {
                "ethnicity": {"caucasian": 0.35, "african_american": 0.35, "hispanic": 0.25, "asian": 0.03, "other": 0.02},
                "income": {"low": 0.50, "lower_middle": 0.30, "middle": 0.15, "upper_middle": 0.04, "high": 0.01},
                "education": {"high_school": 0.50, "college": 0.35, "graduate": 0.15},
                "smoking": {"never": 0.50, "former": 0.25, "current": 0.25},
                "screening_rate": 0.60
            }
        }
        
        logger.info("Initialized patient population generator")
    
    def generate_population_characteristics(
        self,
        site_id: str,
        population_type: str,
        total_patients: int
    ) -> PopulationCharacteristics:
        """Generate population-level characteristics."""
        if population_type not in self.population_templates:
            raise ValueError(f"Unknown population type: {population_type}")
        
        template = self.population_templates[population_type]
        
        # Age distribution (realistic for cancer screening populations)
        age_dist = {
            AgeGroup.PEDIATRIC: 0.05,
            AgeGroup.YOUNG_ADULT: 0.15,
            AgeGroup.MIDDLE_AGED: 0.35,
            AgeGroup.ELDERLY: 0.35,
            AgeGroup.GERIATRIC: 0.10
        }
        
        # Gender distribution
        gender_dist = {"F": 0.52, "M": 0.48}
        
        # Insurance distribution varies by population type
        insurance_base = {
            "urban_diverse": {"private": 0.60, "medicare": 0.20, "medicaid": 0.15, "uninsured": 0.05},
            "suburban_affluent": {"private": 0.80, "medicare": 0.15, "medicaid": 0.03, "uninsured": 0.02},
            "rural_underserved": {"private": 0.45, "medicare": 0.25, "medicaid": 0.20, "uninsured": 0.10},
            "academic_medical": {"private": 0.65, "medicare": 0.20, "medicaid": 0.12, "uninsured": 0.03},
            "safety_net": {"private": 0.30, "medicare": 0.25, "medicaid": 0.35, "uninsured": 0.10}
        }
        
        # Calculate disease prevalence with population-specific adjustments
        prevalence = {}
        for disease, base_rate in self.base_prevalence.items():
            # Adjust for population characteristics
            multiplier = 1.0
            
            # Socioeconomic factors
            if template["income"]["low"] > 0.3:  # High poverty
                multiplier *= 1.2
            
            # Smoking rates
            if template["smoking"]["current"] > 0.2:  # High smoking
                if disease == "lung_cancer":
                    multiplier *= 2.0
            
            # Screening rates
            screening_effect = template["screening_rate"]
            if disease in ["breast_cancer", "colon_cancer"]:
                # Better screening = earlier detection but same incidence
                multiplier *= (0.8 + 0.4 * screening_effect)
            
            prevalence[disease] = base_rate * multiplier
        
        return PopulationCharacteristics(
            site_id=site_id,
            population_name=population_type,
            total_patients=total_patients,
            age_distribution=age_dist,
            gender_distribution=gender_dist,
            ethnicity_distribution=template["ethnicity"],
            insurance_distribution=insurance_base[population_type],
            income_distribution={
                SocioeconomicStatus.LOW: template["income"]["low"],
                SocioeconomicStatus.LOWER_MIDDLE: template["income"]["lower_middle"],
                SocioeconomicStatus.MIDDLE: template["income"]["middle"],
                SocioeconomicStatus.UPPER_MIDDLE: template["income"]["upper_middle"],
                SocioeconomicStatus.HIGH: template["income"]["high"]
            },
            education_distribution=template["education"],
            smoking_rates=template["smoking"],
            alcohol_rates={"none": 0.35, "moderate": 0.50, "heavy": 0.15},
            disease_prevalence=prevalence,
            screening_participation_rate=template["screening_rate"],
            diagnostic_delay_mean_days=30 + (1 - template["screening_rate"]) * 60,
            specialist_access_rate=template["screening_rate"] * 0.9
        )
    
    def generate_patient_cohort(
        self,
        characteristics: PopulationCharacteristics,
        cohort_size: int,
        disease_focus: Optional[str] = None
    ) -> List[PatientProfile]:
        """Generate cohort of individual patients."""
        patients = []
        
        for i in range(cohort_size):
            patient_id = f"{characteristics.site_id}_patient_{i+1:06d}"
            
            # Sample demographics
            age_group = self._sample_from_distribution(characteristics.age_distribution)
            age = self._sample_age_from_group(age_group)
            gender = self._sample_from_distribution(characteristics.gender_distribution)
            ethnicity = self._sample_from_distribution(characteristics.ethnicity_distribution)
            
            # Sample socioeconomic factors
            insurance = self._sample_from_distribution(characteristics.insurance_distribution)
            income = self._sample_from_distribution({k.value: v for k, v in characteristics.income_distribution.items()})
            education = self._sample_from_distribution(characteristics.education_distribution)
            
            # Sample health behaviors
            smoking = self._sample_from_distribution(characteristics.smoking_rates)
            alcohol = self._sample_from_distribution(characteristics.alcohol_rates)
            
            # Sample clinical factors
            family_history = np.random.random() < 0.15  # 15% have family history
            
            # Sample comorbidities (age-dependent)
            diabetes = self._sample_comorbidity("diabetes", age, ethnicity)
            hypertension = self._sample_comorbidity("hypertension", age, ethnicity)
            heart_disease = self._sample_comorbidity("heart_disease", age, gender)
            obesity = self._sample_comorbidity("obesity", age, income)
            immunocompromised = np.random.random() < 0.05  # 5% immunocompromised
            
            # Calculate disease risk scores
            risk_scores = self._calculate_risk_scores(
                age, gender, ethnicity, smoking, family_history,
                diabetes, hypertension, obesity
            )
            
            # Sample presentation characteristics (if disease present)
            symptom_duration = None
            presentation_stage = None
            diagnostic_delay = None
            
            if disease_focus:
                # Simulate disease presentation
                has_disease = np.random.random() < (characteristics.disease_prevalence[disease_focus] / 100000)
                if has_disease:
                    symptom_duration = max(1, int(np.random.exponential(30)))  # Days
                    presentation_stage = self._sample_presentation_stage(age, income, characteristics.screening_participation_rate)
                    diagnostic_delay = max(0, int(np.random.exponential(characteristics.diagnostic_delay_mean_days)))
            
            patient = PatientProfile(
                patient_id=patient_id,
                site_id=characteristics.site_id,
                age=age,
                gender=gender,
                ethnicity=ethnicity,
                insurance_type=insurance,
                income_bracket=SocioeconomicStatus(income),
                education_level=education,
                smoking_history=smoking,
                alcohol_use=alcohol,
                family_history_cancer=family_history,
                diabetes=diabetes,
                hypertension=hypertension,
                heart_disease=heart_disease,
                obesity=obesity,
                immunocompromised=immunocompromised,
                breast_cancer_risk_score=risk_scores["breast_cancer"],
                lung_cancer_risk_score=risk_scores["lung_cancer"],
                prostate_cancer_risk_score=risk_scores["prostate_cancer"],
                colon_cancer_risk_score=risk_scores["colon_cancer"],
                melanoma_risk_score=risk_scores["melanoma"],
                symptom_duration_days=symptom_duration,
                presentation_stage=presentation_stage,
                diagnostic_delay_days=diagnostic_delay
            )
            
            patients.append(patient)
        
        logger.info(f"Generated cohort of {cohort_size} patients for {characteristics.site_id}")
        return patients
    
    def _sample_from_distribution(self, distribution: Dict[str, float]) -> str:
        """Sample from categorical distribution."""
        items = list(distribution.keys())
        probs = list(distribution.values())
        return np.random.choice(items, p=probs)
    
    def _sample_age_from_group(self, age_group: AgeGroup) -> int:
        """Sample age from age group."""
        if age_group == AgeGroup.PEDIATRIC:
            return np.random.randint(0, 18)
        elif age_group == AgeGroup.YOUNG_ADULT:
            return np.random.randint(18, 40)
        elif age_group == AgeGroup.MIDDLE_AGED:
            return np.random.randint(40, 65)
        elif age_group == AgeGroup.ELDERLY:
            return np.random.randint(65, 80)
        else:  # GERIATRIC
            return np.random.randint(80, 95)
    
    def _sample_comorbidity(self, condition: str, age: int, factor: str) -> bool:
        """Sample comorbidity presence based on risk factors."""
        base_rates = {
            "diabetes": 0.10,
            "hypertension": 0.25,
            "heart_disease": 0.08,
            "obesity": 0.35
        }
        
        base_rate = base_rates.get(condition, 0.05)
        
        # Age adjustment
        if age > 65:
            base_rate *= 2.0
        elif age > 50:
            base_rate *= 1.5
        
        # Factor-specific adjustments
        if condition == "diabetes" and factor in ["african_american", "hispanic"]:
            base_rate *= 1.5
        elif condition == "hypertension" and factor == "african_american":
            base_rate *= 1.8
        elif condition == "obesity" and factor == "low":  # Low income
            base_rate *= 1.3
        
        return np.random.random() < min(base_rate, 0.8)  # Cap at 80%
    
    def _calculate_risk_scores(
        self,
        age: int,
        gender: str,
        ethnicity: str,
        smoking: str,
        family_history: bool,
        diabetes: bool,
        hypertension: bool,
        obesity: bool
    ) -> Dict[str, float]:
        """Calculate disease-specific risk scores."""
        scores = {}
        
        for disease in ["breast_cancer", "lung_cancer", "prostate_cancer", "colon_cancer", "melanoma"]:
            base_score = 0.1  # Base 10% risk
            
            # Age factor
            if age > 65:
                base_score *= 3.0
            elif age > 50:
                base_score *= 2.0
            elif age > 40:
                base_score *= 1.5
            
            # Gender-specific diseases
            if disease == "breast_cancer" and gender == "M":
                base_score *= 0.01  # Very rare in men
            elif disease == "prostate_cancer" and gender == "F":
                base_score = 0.0  # Impossible
            
            # Smoking
            if smoking == "current" and disease == "lung_cancer":
                base_score *= 15.0
            elif smoking == "former" and disease == "lung_cancer":
                base_score *= 5.0
            
            # Family history
            if family_history:
                base_score *= self.risk_multipliers["family_history"].get(disease, 1.2)
            
            # Comorbidities
            if obesity and disease in ["breast_cancer", "colon_cancer"]:
                base_score *= 1.3
            
            # Ethnicity factors
            if ethnicity == "african_american":
                if disease == "prostate_cancer":
                    base_score *= 2.0
                elif disease == "breast_cancer":
                    base_score *= 1.3
            elif ethnicity == "caucasian" and disease == "melanoma":
                base_score *= 3.0
            
            # Cap at reasonable maximum
            scores[disease] = min(base_score, 0.8)
        
        return scores
    
    def _sample_presentation_stage(self, age: int, income: str, screening_rate: float) -> str:
        """Sample disease presentation stage."""
        # Better screening and higher income → earlier stage
        early_prob = 0.4 + 0.3 * screening_rate
        if income in ["upper_middle", "high"]:
            early_prob += 0.1
        if age > 65:  # Medicare screening
            early_prob += 0.1
        
        rand = np.random.random()
        if rand < early_prob:
            return "early"
        elif rand < early_prob + 0.4:
            return "intermediate"
        else:
            return "advanced"
    
    def analyze_population_diversity(self, patients: List[PatientProfile]) -> Dict[str, Any]:
        """Analyze diversity metrics of patient population."""
        if not patients:
            return {"error": "No patients provided"}
        
        # Convert to DataFrame for analysis
        data = []
        for p in patients:
            data.append({
                "age": p.age,
                "gender": p.gender,
                "ethnicity": p.ethnicity,
                "income": p.income_bracket.value,
                "insurance": p.insurance_type,
                "smoking": p.smoking_history,
                "diabetes": p.diabetes,
                "hypertension": p.hypertension,
                "obesity": p.obesity
            })
        
        df = pd.DataFrame(data)
        
        # Calculate diversity metrics
        analysis = {
            "total_patients": len(patients),
            "demographics": {
                "age_stats": {
                    "mean": df["age"].mean(),
                    "std": df["age"].std(),
                    "min": df["age"].min(),
                    "max": df["age"].max()
                },
                "gender_distribution": df["gender"].value_counts(normalize=True).to_dict(),
                "ethnicity_distribution": df["ethnicity"].value_counts(normalize=True).to_dict()
            },
            "socioeconomic": {
                "income_distribution": df["income"].value_counts(normalize=True).to_dict(),
                "insurance_distribution": df["insurance"].value_counts(normalize=True).to_dict()
            },
            "health_behaviors": {
                "smoking_distribution": df["smoking"].value_counts(normalize=True).to_dict()
            },
            "comorbidities": {
                "diabetes_rate": df["diabetes"].mean(),
                "hypertension_rate": df["hypertension"].mean(),
                "obesity_rate": df["obesity"].mean()
            },
            "risk_scores": {
                "breast_cancer_mean": np.mean([p.breast_cancer_risk_score for p in patients]),
                "lung_cancer_mean": np.mean([p.lung_cancer_risk_score for p in patients]),
                "prostate_cancer_mean": np.mean([p.prostate_cancer_risk_score for p in patients]),
                "colon_cancer_mean": np.mean([p.colon_cancer_risk_score for p in patients]),
                "melanoma_mean": np.mean([p.melanoma_risk_score for p in patients])
            }
        }
        
        return analysis
    
    def export_population(self, patients: List[PatientProfile], filepath: str) -> None:
        """Export patient population to JSON file."""
        patients_data = []
        
        for patient in patients:
            patient_dict = asdict(patient)
            # Convert enums to strings
            patient_dict["income_bracket"] = patient.income_bracket.value
            patients_data.append(patient_dict)
        
        with open(filepath, 'w') as f:
            json.dump(patients_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(patients)} patients to {filepath}")


if __name__ == "__main__":
    # Demo: Patient population modeling
    
    print("=== Patient Population Modeling Demo ===\n")
    
    # Create generator
    generator = PatientPopulationGenerator(random_seed=42)
    
    # Generate different population types
    population_types = [
        ("urban_diverse", "Urban Diverse"),
        ("suburban_affluent", "Suburban Affluent"), 
        ("rural_underserved", "Rural Underserved"),
        ("academic_medical", "Academic Medical"),
        ("safety_net", "Safety Net")
    ]
    
    for pop_type, pop_name in population_types:
        print(f"--- {pop_name} Population ---")
        
        # Generate population characteristics
        characteristics = generator.generate_population_characteristics(
            site_id=f"site_{pop_type}",
            population_type=pop_type,
            total_patients=10000
        )
        
        print(f"Screening rate: {characteristics.screening_participation_rate:.1%}")
        print(f"Diagnostic delay: {characteristics.diagnostic_delay_mean_days:.0f} days")
        
        # Generate patient cohort
        patients = generator.generate_patient_cohort(
            characteristics=characteristics,
            cohort_size=500,
            disease_focus="lung_cancer"
        )
        
        # Analyze diversity
        analysis = generator.analyze_population_diversity(patients)
        
        print(f"Age: {analysis['demographics']['age_stats']['mean']:.1f} ± {analysis['demographics']['age_stats']['std']:.1f}")
        print(f"Gender: {analysis['demographics']['gender_distribution']}")
        print(f"Ethnicity: {list(analysis['demographics']['ethnicity_distribution'].keys())[:3]}")
        print(f"Smoking rate: {analysis['health_behaviors']['smoking_distribution'].get('current', 0):.1%}")
        print(f"Diabetes rate: {analysis['comorbidities']['diabetes_rate']:.1%}")
        print(f"Lung cancer risk: {analysis['risk_scores']['lung_cancer_mean']:.3f}")
        print()
    
    # Export example population
    generator.export_population(patients, "patient_population_example.json")
    print("Patient population exported to patient_population_example.json")
    
    print("\n=== Demo Complete ===")
