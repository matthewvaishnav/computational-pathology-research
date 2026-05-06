#!/usr/bin/env python3
"""
PathologyFL Validation - Validate medical expertise assumptions
"""

from pathology_fl_demo import PathologyFLDemo, HospitalMetadata, SlideQuality, HospitalType, CancerType

def validate_medical_assumptions():
    """Validate PathologyFL medical expertise assumptions."""
    
    print("🏥 PathologyFL Medical Validation")
    print("=" * 50)
    
    demo = PathologyFLDemo()
    validations = []
    
    # Validation 1: Cancer centers > Teaching hospitals
    cancer_center = HospitalMetadata(
        hospital_id="cancer_center",
        hospital_type=HospitalType.CANCER_CENTER,
        annual_cases=10000,
        cancer_specialties=[CancerType.BREAST],
        diagnostic_accuracy=0.94,
        years_experience=15
    )
    
    teaching_hospital = HospitalMetadata(
        hospital_id="teaching_hospital",
        hospital_type=HospitalType.TEACHING_HOSPITAL,
        annual_cases=10000,
        cancer_specialties=[CancerType.BREAST],
        diagnostic_accuracy=0.94,
        years_experience=15
    )
    
    cancer_weight = demo.calculate_expertise_weight(cancer_center, CancerType.BREAST)
    teaching_weight = demo.calculate_expertise_weight(teaching_hospital, CancerType.BREAST)
    
    validations.append({
        "test": "Cancer centers > Teaching hospitals",
        "passed": cancer_weight > teaching_weight,
        "details": f"{cancer_weight:.3f} > {teaching_weight:.3f}"
    })
    
    # Validation 2: Specialists > Generalists for their specialty
    breast_specialist = HospitalMetadata(
        hospital_id="breast_specialist",
        hospital_type=HospitalType.CANCER_CENTER,
        annual_cases=8000,
        cancer_specialties=[CancerType.BREAST],
        diagnostic_accuracy=0.93,
        years_experience=12
    )
    
    generalist = HospitalMetadata(
        hospital_id="generalist",
        hospital_type=HospitalType.CANCER_CENTER,
        annual_cases=8000,
        cancer_specialties=[CancerType.GENERAL],
        diagnostic_accuracy=0.93,
        years_experience=12
    )
    
    specialist_weight = demo.calculate_expertise_weight(breast_specialist, CancerType.BREAST)
    generalist_weight = demo.calculate_expertise_weight(generalist, CancerType.BREAST)
    
    validations.append({
        "test": "Specialists > Generalists for specialty",
        "passed": specialist_weight > generalist_weight,
        "details": f"{specialist_weight:.3f} > {generalist_weight:.3f}"
    })
    
    # Validation 3: Higher volume increases weight
    high_volume = HospitalMetadata(
        hospital_id="high_volume",
        hospital_type=HospitalType.COMMUNITY_HOSPITAL,
        annual_cases=15000,
        cancer_specialties=[CancerType.GENERAL],
        diagnostic_accuracy=0.85,
        years_experience=10
    )
    
    low_volume = HospitalMetadata(
        hospital_id="low_volume",
        hospital_type=HospitalType.COMMUNITY_HOSPITAL,
        annual_cases=1000,
        cancer_specialties=[CancerType.GENERAL],
        diagnostic_accuracy=0.85,
        years_experience=10
    )
    
    high_vol_weight = demo.calculate_expertise_weight(high_volume, CancerType.GENERAL)
    low_vol_weight = demo.calculate_expertise_weight(low_volume, CancerType.GENERAL)
    
    validations.append({
        "test": "Higher volume increases weight",
        "passed": high_vol_weight > low_vol_weight,
        "details": f"{high_vol_weight:.3f} > {low_vol_weight:.3f}"
    })
    
    # Validation 4: Quality affects weighting
    high_quality = SlideQuality(0.9, 0.85, 0.92, 0.1)
    low_quality = SlideQuality(0.6, 0.55, 0.65, 0.4)
    
    high_qual_weight = demo.calculate_quality_weight(high_quality)
    low_qual_weight = demo.calculate_quality_weight(low_quality)
    
    validations.append({
        "test": "Higher quality increases weight",
        "passed": high_qual_weight > low_qual_weight,
        "details": f"{high_qual_weight:.3f} > {low_qual_weight:.3f}"
    })
    
    # Results
    print("Medical Expertise Validation Results:")
    print("-" * 50)
    
    passed = 0
    for validation in validations:
        status = "✅ PASS" if validation["passed"] else "❌ FAIL"
        print(f"{status} {validation['test']}")
        print(f"     {validation['details']}")
        if validation["passed"]:
            passed += 1
    
    print(f"\nValidation Summary: {passed}/{len(validations)} passed")
    
    if passed == len(validations):
        print("🏆 All medical expertise assumptions validated!")
    else:
        print("⚠️ Some medical assumptions need review")
    
    return passed == len(validations)

if __name__ == "__main__":
    validate_medical_assumptions()