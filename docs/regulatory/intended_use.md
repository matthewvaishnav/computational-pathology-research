# Intended Use Statement - HistoCore AI Pathology Analysis System

**Document Version:** 1.0  
**Date:** January 2025  
**Regulatory Reference:** FDA 510(k) Premarket Notification  
**Document ID:** REG-IU-001-v1.0

## 1. Intended Use Statement

**HistoCore AI Pathology Analysis System is intended for use by qualified pathologists as a diagnostic aid in the analysis of digitized whole slide images of histopathology specimens to assist in the identification and classification of cancer in breast, lung, prostate, colon, and melanoma tissue specimens.**

## 2. Detailed Indications for Use

### 2.1 Primary Indications

HistoCore is indicated for use as a diagnostic aid for qualified pathologists in the analysis of whole slide images from formalin-fixed, paraffin-embedded (FFPE) tissue specimens for the following cancer types:

#### 2.1.1 Breast Cancer Analysis
**Specimen Types:**
- Core needle biopsies
- Surgical excision specimens
- Mastectomy specimens
- Sentinel lymph node biopsies

**Diagnostic Capabilities:**
- Detection of invasive ductal carcinoma (IDC)
- Detection of invasive lobular carcinoma (ILC)
- Identification of ductal carcinoma in situ (DCIS)
- Histologic grade assessment using Nottingham grading system (Grades 1-3)
- Lymphovascular invasion detection
- Perineural invasion assessment

**Clinical Applications:**
- Primary diagnosis confirmation
- Grade assessment for treatment planning
- Prognostic factor evaluation
- Quality assurance for diagnostic consistency

#### 2.1.2 Lung Cancer Analysis
**Specimen Types:**
- Transbronchial biopsies
- CT-guided needle biopsies
- Surgical resection specimens
- Pleural biopsies

**Diagnostic Capabilities:**
- Adenocarcinoma detection and subtyping
- Squamous cell carcinoma identification
- Small cell vs. non-small cell lung cancer differentiation
- Neuroendocrine tumor detection
- Tumor staging assessment (T1-T4)

**Clinical Applications:**
- Primary diagnosis establishment
- Histologic subtyping for targeted therapy selection
- Staging assessment for treatment planning
- Differential diagnosis support

#### 2.1.3 Prostate Cancer Analysis
**Specimen Types:**
- Transrectal ultrasound (TRUS) guided biopsies
- Transperineal biopsies
- Radical prostatectomy specimens
- Transurethral resection specimens

**Diagnostic Capabilities:**
- Adenocarcinoma detection
- Gleason score assessment (Grade Groups 1-5)
- Perineural invasion detection
- Extraprostatic extension assessment
- Seminal vesicle invasion evaluation

**Clinical Applications:**
- Primary diagnosis confirmation
- Gleason grading for risk stratification
- Prognostic assessment
- Treatment planning support

#### 2.1.4 Colon Cancer Analysis
**Specimen Types:**
- Colonoscopic biopsies
- Surgical resection specimens
- Polypectomy specimens
- Endoscopic mucosal resection specimens

**Diagnostic Capabilities:**
- Adenocarcinoma detection and grading
- Tumor staging assessment (T1-T4)
- Lymphovascular invasion detection
- Perineural invasion assessment
- Microsatellite instability morphologic features

**Clinical Applications:**
- Primary diagnosis establishment
- Staging for treatment planning
- Prognostic factor assessment
- Surveillance biopsy evaluation

#### 2.1.5 Melanoma Analysis
**Specimen Types:**
- Skin punch biopsies
- Shave biopsies
- Excision specimens
- Sentinel lymph node biopsies

**Diagnostic Capabilities:**
- Melanoma detection and classification
- Breslow depth measurement
- Clark level determination
- Ulceration assessment
- Mitotic rate evaluation

**Clinical Applications:**
- Primary diagnosis confirmation
- Staging parameter assessment
- Prognostic evaluation
- Treatment planning support

### 2.2 User Qualifications

#### 2.2.1 Primary Users
**Qualified Pathologists:**
- Board-certified anatomic pathologists
- Pathology residents under supervision
- Pathologists with subspecialty training in relevant areas

**Required Qualifications:**
- Medical degree (MD, DO, or equivalent)
- Completed pathology residency training
- Board certification in anatomic pathology
- Current medical license in good standing
- Completion of HistoCore training and certification program

#### 2.2.2 Secondary Users
**Laboratory Personnel:**
- Medical technologists with histology certification
- Pathology assistants with appropriate training
- Laboratory supervisors and managers

**Usage Limitations:**
- Secondary users may operate the system under pathologist supervision
- Final diagnostic interpretation must be performed by qualified pathologist
- System results require pathologist review and approval

### 2.3 Clinical Environment

#### 2.3.1 Approved Settings
- Hospital pathology departments
- Independent pathology laboratories
- Academic medical centers
- Specialty cancer centers
- Telepathology consultation services

#### 2.3.2 Infrastructure Requirements
- Digital pathology infrastructure with whole slide imaging capability
- Appropriate IT security and HIPAA compliance measures
- Quality assurance programs for digital pathology
- Trained technical support personnel

## 3. Contraindications

### 3.1 Absolute Contraindications
1. **Frozen Section Analysis**
   - System not validated for intraoperative frozen section diagnosis
   - Different tissue preparation affects algorithm performance

2. **Non-FFPE Specimens**
   - Cytology preparations not supported
   - Fresh tissue specimens not validated
   - Decalcified specimens may have altered morphology

3. **Unsupported Tissue Types**
   - Hematologic malignancies (lymphomas, leukemias)
   - Sarcomas and mesenchymal tumors
   - Central nervous system tumors
   - Pediatric tumors with unique morphology

### 3.2 Relative Contraindications
1. **Poor Image Quality**
   - Significant focus artifacts
   - Severe compression artifacts
   - Inadequate resolution (<0.5 μm/pixel)

2. **Specimen Preparation Issues**
   - Severe fixation artifacts
   - Inadequate staining quality
   - Significant tissue folding or tears

3. **Unusual Morphologic Variants**
   - Rare histologic subtypes not in training data
   - Unusual staining patterns
   - Significant treatment-related changes

## 4. Warnings and Precautions

### 4.1 Critical Warnings

#### 4.1.1 Diagnostic Limitations
**⚠️ WARNING: This device is not intended to replace pathologist judgment or serve as the sole basis for diagnosis. All AI-generated results must be reviewed, interpreted, and approved by a qualified pathologist.**

**⚠️ WARNING: The system may not detect all instances of cancer, particularly in cases with:**
- Minimal or focal disease
- Unusual morphologic variants
- Poor image quality
- Artifacts from specimen preparation

**⚠️ WARNING: False positive results may occur, leading to unnecessary anxiety or treatment. Always correlate AI results with clinical history and other diagnostic findings.**

#### 4.1.2 User Training Requirements
**⚠️ WARNING: Users must complete mandatory training and certification before clinical use. Inadequate training may result in misinterpretation of results or inappropriate clinical decisions.**

### 4.2 Important Precautions

#### 4.2.1 System Performance
- **Performance Monitoring:** Regularly monitor system performance metrics and report any significant changes
- **Quality Control:** Implement daily quality control procedures using reference slides
- **Calibration:** Ensure proper system calibration and maintenance per manufacturer specifications

#### 4.2.2 Clinical Integration
- **Workflow Integration:** Ensure proper integration with existing laboratory workflows and information systems
- **Result Documentation:** Maintain appropriate documentation of AI results and pathologist interpretation
- **Audit Trail:** Preserve complete audit trail of system usage and diagnostic decisions

#### 4.2.3 Technical Considerations
- **Image Quality:** Verify adequate image quality before analysis
- **System Updates:** Install security updates and software patches promptly
- **Backup Procedures:** Maintain backup procedures for system failures

## 5. Performance Limitations

### 5.1 Analytical Limitations

#### 5.1.1 Sensitivity Limitations
- System sensitivity may be reduced in cases with:
  - Minimal residual disease after treatment
  - Small focus of cancer (<1mm)
  - Poorly differentiated tumors with unusual morphology
  - Significant background inflammation or necrosis

#### 5.1.2 Specificity Limitations
- False positive results may occur with:
  - Reactive/inflammatory conditions mimicking malignancy
  - Benign lesions with atypical features
  - Artifacts from tissue processing
  - Unusual staining patterns

#### 5.1.3 Quantitative Limitations
- Measurement accuracy may be affected by:
  - Image resolution and calibration
  - Tissue thickness variations
  - Staining intensity variations
  - Compression artifacts

### 5.2 Technical Limitations

#### 5.2.1 Image Requirements
- **Resolution:** Minimum 0.25 μm/pixel (40x equivalent)
- **File Format:** Support for standard WSI formats (SVS, NDPI, CZI, etc.)
- **Color Space:** RGB color images with standard H&E staining
- **Compression:** Minimal compression artifacts (>90% JPEG quality)

#### 5.2.2 Processing Limitations
- **Slide Size:** Maximum 200,000 x 200,000 pixels
- **Processing Time:** May exceed 30 seconds for very large slides
- **Concurrent Processing:** Limited by available computational resources
- **Network Dependency:** Requires stable network connection for cloud-based processing

## 6. Clinical Validation Summary

### 6.1 Validation Studies

#### 6.1.1 Multi-Site Clinical Study
- **Study Design:** Prospective, multi-center validation study
- **Sample Size:** 10,000 cases across 15 institutions
- **Reference Standard:** Consensus diagnosis by expert pathologists
- **Primary Endpoint:** Sensitivity and specificity for cancer detection

#### 6.1.2 Performance Results
| Cancer Type | Cases (n) | Sensitivity | Specificity | PPV | NPV |
|-------------|-----------|-------------|-------------|-----|-----|
| Breast | 2,500 | 94.2% | 96.8% | 95.1% | 96.2% |
| Lung | 2,000 | 92.7% | 95.4% | 93.8% | 94.7% |
| Prostate | 2,500 | 93.8% | 94.9% | 94.2% | 94.6% |
| Colon | 2,000 | 91.5% | 96.2% | 94.7% | 94.1% |
| Melanoma | 1,000 | 95.1% | 97.3% | 96.2% | 96.5% |

#### 6.1.3 Subgroup Analysis
- **Scanner Variability:** Consistent performance across 5 major scanner manufacturers
- **Institutional Variability:** <3% performance variation across sites
- **Demographic Subgroups:** Equivalent performance across age, sex, and ethnicity groups

### 6.2 Real-World Evidence

#### 6.2.1 Post-Market Studies
- **Deployment Sites:** 25+ hospitals and laboratories
- **Cases Processed:** >50,000 clinical cases
- **User Satisfaction:** 92% pathologist satisfaction rate
- **Clinical Impact:** 15% reduction in diagnostic turnaround time

#### 6.2.2 Continuous Monitoring
- **Performance Tracking:** Real-time monitoring of diagnostic accuracy
- **Drift Detection:** Automated detection of performance degradation
- **User Feedback:** Systematic collection and analysis of user reports

## 7. Regulatory Compliance

### 7.1 FDA Regulatory Status
- **Regulatory Pathway:** 510(k) Premarket Notification
- **Device Classification:** Class II Medical Device Software
- **Product Code:** LLZ (System, Image Processing, Radiological)
- **Predicate Devices:** PathAI AISight (K193658), Paige Prostate (K212563)

### 7.2 Quality System Compliance
- **ISO 13485:2016:** Medical devices quality management system
- **IEC 62304:2006:** Medical device software lifecycle processes
- **ISO 14971:2019:** Medical devices risk management
- **IEC 62366-1:2015:** Medical devices usability engineering

### 7.3 International Compliance
- **CE Marking:** European Conformity marking for EU market
- **Health Canada:** Medical Device License for Canadian market
- **TGA Australia:** Therapeutic Goods Administration approval
- **PMDA Japan:** Pharmaceuticals and Medical Devices Agency consultation

## 8. Post-Market Commitments

### 8.1 Surveillance Activities
- **Adverse Event Monitoring:** Continuous monitoring and reporting per 21 CFR 803
- **Performance Monitoring:** Real-time tracking of diagnostic performance metrics
- **User Training:** Ongoing education and competency assessment programs

### 8.2 Product Improvements
- **Software Updates:** Regular updates to improve performance and add features
- **Model Retraining:** Periodic retraining with new data to maintain accuracy
- **User Feedback Integration:** Systematic incorporation of user feedback into product development

### 8.3 Clinical Studies
- **Long-term Outcomes:** Studies tracking patient outcomes and clinical impact
- **Comparative Effectiveness:** Studies comparing AI-assisted vs. traditional diagnosis
- **Health Economics:** Analysis of cost-effectiveness and healthcare utilization

## 9. Conclusion

The intended use of HistoCore AI Pathology Analysis System is clearly defined to provide diagnostic assistance to qualified pathologists in the analysis of specific cancer types from digitized histopathology specimens. The system's validation demonstrates substantial equivalence to predicate devices while offering enhanced capabilities through advanced AI technology.

The comprehensive warnings, precautions, and limitations ensure appropriate clinical use while maximizing the system's potential to improve diagnostic accuracy and efficiency in pathology practice.

---

**Document Control:**
- **Author:** HistoCore Regulatory Affairs Team
- **Clinical Reviewer:** Chief Medical Officer
- **Technical Reviewer:** VP of Engineering
- **Regulatory Reviewer:** Director of Regulatory Affairs
- **Approver:** Chief Executive Officer
- **Next Review Date:** January 2026
- **Document ID:** REG-IU-001-v1.0