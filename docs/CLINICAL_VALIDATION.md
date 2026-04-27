# Clinical Validation Framework

Multi-site validation system. Synthetic hospital environments simulate real-world deployment conditions.

## Synthetic Sites (`src/clinical_validation/synthetic_sites.py`)

Generate 5 diverse hospital types:
- **Academic Medical Centers**: Premium equipment, high data quality, research focus
- **Community Hospitals**: Standard equipment, moderate volume, general population  
- **Regional Medical Centers**: Mixed characteristics, broad catchment area
- **Specialty Cancer Centers**: Premium equipment, expert staff, cancer focus
- **Rural Hospitals**: Basic equipment, limited resources, underserved populations

## Site Characteristics

### Patient Demographics
- Age/gender distribution by region
- Ethnicity (Caucasian, African American, Hispanic, Asian)
- Socioeconomic factors (insurance, income)
- Disease prevalence rates per 100K population

### Data Quality Profiles
- Image resolution + quality scores
- Staining consistency + compression artifacts
- Annotation quality (expert vs resident)
- Scanner calibration + color consistency

### Operational Profiles  
- Staffing levels (pathologists, residents, technicians)
- Case volume + turnaround times
- Technology adoption (digital pathology, AI familiarity)
- Quality metrics (error rates, second opinions)

## Validation Datasets

Each site generates realistic validation data:
- Case distribution based on demographics
- Ground truth availability (50-100%)
- Quality characteristics matching site profile
- Disease-specific prevalence patterns

## Geographic Diversity

5 US regions with distinct characteristics:
- **Northeast**: High income, diverse, academic centers
- **Southeast**: Mixed demographics, moderate income
- **Midwest**: Predominantly Caucasian, industrial
- **West**: Hispanic majority, high income, tech-forward
- **Southwest**: Hispanic majority, mixed income

## Usage

```python
# Generate synthetic sites
generator = SyntheticSiteGenerator()
sites = generator.generate_sites(count=5)

# Create validation datasets
for site in sites:
    dataset = generator.generate_validation_dataset(site)
    print(f"{site.site_name}: {dataset['total_cases']} cases")
```

## Validation Metrics

- **Site diversity**: 5 hospital types across 5 regions
- **Patient diversity**: Realistic demographic distributions
- **Data quality range**: 0.65-0.95 quality scores
- **Case volume**: 500-30K annual cases per site
- **Ground truth**: 50-100% availability per site

## Data Quality Simulation (`src/clinical_validation/data_quality_simulation.py`)

Comprehensive data quality simulation system modeling realistic conditions across hospital environments.

### Quality Profiles

Five distinct hospital quality profiles:

1. **Academic Medical Center** - Excellent quality (noise: 5%, blur: 2%)
2. **Large Community Hospital** - Good quality (noise: 10%, blur: 5%)
3. **Regional Hospital** - Fair quality (noise: 18%, blur: 12%)
4. **Rural Hospital** - Poor quality (noise: 25%, blur: 20%)
5. **Resource Limited** - Critical quality (noise: 35%, blur: 30%)

### Quality Degradation Types

- **Noise Addition**: Gaussian noise with spatial correlation
- **Blur Simulation**: Focus issues and motion blur
- **Compression Artifacts**: JPEG compression simulation
- **Color Shift**: Staining variation simulation
- **Scanner Artifacts**: Equipment-specific degradation patterns

### Usage Example

```python
from src.clinical_validation.data_quality_simulation import DataQualitySimulator

simulator = DataQualitySimulator(seed=42)
profile = simulator.get_site_profile("rural_hospital")
degraded_image = simulator.simulate_image_quality_degradation(image, profile)
```

## Realistic Noise Patterns (`src/clinical_validation/noise_patterns.py`)

Advanced noise pattern generation for histopathology images with 10+ noise types.

### Noise Types Supported

- **Gaussian**: Basic sensor noise with spatial correlation
- **Poisson**: Photon shot noise
- **Salt & Pepper**: Random pixel corruption
- **Speckle**: Multiplicative noise
- **Periodic**: Scanner line artifacts
- **Impulse**: Random bright/dark spots
- **Quantization**: Bit depth reduction effects
- **Thermal**: Temperature-dependent noise
- **Shot**: Photon counting noise
- **Readout**: CCD/CMOS sensor noise

### Scanner-Specific Profiles

Different scanners have characteristic noise patterns:
- **Leica Aperio GT450**: Low noise, minimal artifacts
- **Hamamatsu NanoZoomer**: Moderate noise, periodic patterns
- **Legacy Scanners**: High noise, multiple artifact types

### Usage Example

```python
from src.clinical_validation.noise_patterns import RealisticNoiseGenerator

noise_gen = RealisticNoiseGenerator(seed=42)
noisy_image = noise_gen.apply_scanner_noise(image, "Legacy_Scanner", severity=1.5)
```

## Cross-Validation Strategy (`src/clinical_validation/cross_validation_strategy.py`)

Sophisticated cross-validation strategies for medical AI validation.

### Validation Strategies

- **Stratified K-Fold**: Standard stratified splits
- **Site-Stratified**: Ensures site balance across folds
- **Patient-Stratified**: Prevents patient leakage
- **Leave-One-Site-Out**: Tests generalization to new sites
- **Temporal Split**: Time-series validation
- **Nested CV**: Hyperparameter tuning with validation
- **Monte Carlo CV**: Random repeated splits
- **Bootstrap**: Bootstrap sampling validation

### Key Features

- **Patient Leakage Prevention**: Ensures no patient appears in both train/test
- **Site Balance**: Maintains site representation across folds
- **Class Stratification**: Preserves disease distribution
- **Quality Validation**: Comprehensive split quality assessment

### Usage Example

```python
from src.clinical_validation.cross_validation_strategy import (
    ClinicalCrossValidator, ValidationConfig, ValidationStrategy
)

config = ValidationConfig(
    strategy=ValidationStrategy.SITE_STRATIFIED,
    n_folds=5,
    balance_sites=True
)

validator = ClinicalCrossValidator(config)
folds = validator.create_validation_splits(data)
report = validator.validate_splits(folds, data, 'label', 'site_id')
```