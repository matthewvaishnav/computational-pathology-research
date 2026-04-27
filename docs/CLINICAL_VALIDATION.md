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