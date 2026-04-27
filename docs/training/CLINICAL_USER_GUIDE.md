# HistoCore Real-Time WSI Streaming - Clinical User Guide

**For Pathologists, Clinicians, and Clinical Staff**

## 🎯 Quick Start (5 Minutes)

### What This System Does
Analyzes gigapixel pathology slides in **under 30 seconds** with real-time visualization of attention patterns and confidence scores.

### Your First Slide Analysis

1. **Login** with your hospital credentials
2. **Select a slide** from PACS or upload directly
3. **Click "Process"** - watch real-time analysis
4. **Review results** - attention heatmap + confidence score
5. **Generate report** - PDF with visualizations

**That's it.** No technical knowledge required.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Processing Slides](#processing-slides)
3. [Understanding Results](#understanding-results)
4. [Clinical Reports](#clinical-reports)
5. [PACS Integration](#pacs-integration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Getting Started

### System Access

**Web Interface**: `https://histocore.yourhospital.org`

**Login Credentials**: Use your hospital SSO (Single Sign-On)
- Epic/Cerner credentials work automatically
- Contact IT if you need access

**Supported Browsers**:
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

### User Roles

| Role | What You Can Do |
|------|-----------------|
| **Pathologist** | Full access - process, review, approve, export |
| **Clinician** | Process slides, view results, export reports |
| **Technician** | Process slides, view results |
| **Viewer** | View slides and results only |

---

## Processing Slides

### Method 1: From PACS (Recommended)

1. **Open Dashboard** → Click "Process from PACS"
2. **Search for Patient**:
   - Enter MRN, name, or accession number
   - Select study from worklist
3. **Select Slide(s)**:
   - Choose one or multiple slides
   - System shows slide metadata (size, magnification)
4. **Click "Start Processing"**
5. **Watch Real-Time Progress**:
   - Processing bar shows completion %
   - Attention heatmap updates live
   - Confidence score increases progressively

**Processing Time**: 15-30 seconds for typical slides

### Method 2: Direct Upload

1. **Open Dashboard** → Click "Upload Slide"
2. **Select File**:
   - Supported formats: .svs, .tiff, .ndpi, DICOM
   - Max size: 10GB per slide
3. **Enter Metadata** (optional):
   - Patient ID, case number, diagnosis
4. **Click "Upload & Process"**

### Batch Processing

Process multiple slides simultaneously:

1. **Select Multiple Slides** (up to 10)
2. **Click "Batch Process"**
3. **Monitor Progress** in batch queue
4. **Review Results** as each completes

**Concurrent Limit**: 10 slides (system auto-queues additional)

---

## Understanding Results

### Attention Heatmap

**What It Shows**: Areas the AI focused on during analysis

**Color Scale**:
- 🔴 **Red/Hot**: High attention (important regions)
- 🟡 **Yellow**: Moderate attention
- 🔵 **Blue/Cool**: Low attention (background)

**How to Interpret**:
- Red regions = AI found these areas diagnostically significant
- Multiple red regions = heterogeneous features
- Uniform blue = mostly normal tissue

**Clinical Use**:
- Guides your review to regions of interest
- Quality check for AI analysis
- Teaching tool for trainees

### Confidence Score

**Range**: 0-100%

**Interpretation**:
- **90-100%**: High confidence - AI is very certain
- **70-89%**: Moderate confidence - review carefully
- **<70%**: Low confidence - manual review recommended

**What Affects Confidence**:
- Slide quality (focus, staining)
- Tissue type complexity
- Artifact presence
- Amount of diagnostic tissue

**Clinical Decision**:
- High confidence ≠ automatic acceptance
- Always review attention heatmap
- Use as second opinion, not replacement

### Prediction Results

**Classification Output**:
- Primary diagnosis
- Confidence percentage
- Top 3 differential diagnoses (if applicable)

**Quantitative Metrics** (if available):
- Tumor percentage
- Mitotic count
- Ki-67 index
- Other biomarkers

---

## Clinical Reports

### Generating Reports

1. **Review Results** on dashboard
2. **Click "Generate Report"**
3. **Select Template**:
   - Standard diagnostic report
   - Research report (de-identified)
   - Teaching case report
4. **Customize** (optional):
   - Add clinical notes
   - Include/exclude sections
   - Add institutional logo
5. **Click "Generate PDF"**

**Report Contents**:
- Patient demographics (if applicable)
- Slide metadata (size, magnification, staining)
- AI prediction with confidence
- Attention heatmap visualization
- Processing statistics
- Pathologist review section (for sign-off)
- Timestamp and audit trail

### Report Templates

**Standard Diagnostic**:
- Full patient information
- Clinical context
- AI findings
- Pathologist interpretation
- Signature block

**Research Report**:
- De-identified patient data
- Detailed AI metrics
- Statistical analysis
- No PHI included

**Teaching Case**:
- Educational annotations
- Differential diagnosis discussion
- Learning objectives
- References

### Exporting Reports

**Formats**:
- PDF (recommended)
- DOCX (editable)
- JSON (for EMR integration)

**Delivery Options**:
- Download directly
- Send to PACS
- Email to clinician
- Export to EMR

---

## PACS Integration

### Worklist Integration

**Automatic Case Retrieval**:
- System pulls cases from your PACS worklist
- Filters by modality (WSI, digital pathology)
- Shows pending cases requiring review

**Processing from Worklist**:
1. **View Worklist** → Shows all pending cases
2. **Select Case** → Click to open
3. **Process** → One-click analysis
4. **Complete** → Results sent back to PACS

### Result Delivery

**Automatic PACS Upload**:
- Results sent to PACS after processing
- Includes attention heatmap overlay
- Adds structured report to study

**DICOM Structured Report**:
- Standard DICOM SR format
- Compatible with all PACS systems
- Includes AI confidence and findings

---

## Best Practices

### For Accurate Results

✅ **DO**:
- Use high-quality scans (focus, staining)
- Process entire slide (not regions)
- Review attention heatmap carefully
- Compare with your own assessment
- Document AI-assisted diagnosis in report

❌ **DON'T**:
- Rely solely on AI prediction
- Process poor-quality slides
- Skip manual review
- Use for unsupported tissue types
- Share patient data outside system

### Quality Control

**Before Processing**:
- Check slide quality (focus, artifacts)
- Verify correct patient/case information
- Ensure proper staining protocol

**After Processing**:
- Review attention heatmap for sanity
- Check confidence score
- Compare with clinical context
- Document any discrepancies

### Clinical Workflow Integration

**Frozen Section**:
- Process while preparing next case
- Use as preliminary assessment
- Confirm with microscopy

**Routine Diagnosis**:
- Process overnight batch
- Review results in morning
- Prioritize high-confidence cases

**Consultation**:
- Process before teleconsultation
- Share attention heatmap with consultant
- Use as discussion tool

---

## Troubleshooting

### Common Issues

**"Slide Not Found in PACS"**
- Verify accession number
- Check if slide is scanned
- Contact PACS administrator

**"Processing Failed"**
- Check slide format compatibility
- Verify file not corrupted
- Try re-uploading

**"Low Confidence Score"**
- Review slide quality
- Check for artifacts
- Consider manual review

**"Slow Processing"**
- System may be under high load
- Check network connection
- Contact IT if persistent

### Getting Help

**Technical Support**:
- Email: support@histocore.ai
- Phone: 1-800-HISTOCORE
- Hours: 24/7

**Clinical Questions**:
- Email: clinical@histocore.ai
- Response time: <4 hours

**Emergency**:
- Call hospital IT helpdesk
- Reference: "HistoCore AI System"

---

## FAQ

### General

**Q: Does this replace pathologists?**  
A: No. This is a decision support tool, not a replacement. Final diagnosis is always made by a pathologist.

**Q: How accurate is the AI?**  
A: Accuracy varies by tissue type. Validation studies show 85-95% concordance with expert pathologists. Always review results.

**Q: Is my patient data secure?**  
A: Yes. Full HIPAA compliance with encryption, audit trails, and access controls.

**Q: Can I use this for research?**  
A: Yes, with IRB approval. Use de-identified reports.

### Processing

**Q: How long does processing take?**  
A: 15-30 seconds for typical slides (100K-200K patches).

**Q: Can I process multiple slides at once?**  
A: Yes, up to 10 concurrent slides. Additional slides are queued.

**Q: What slide formats are supported?**  
A: .svs, .tiff, .ndpi, DICOM WSI. Contact support for other formats.

**Q: What if processing fails?**  
A: System automatically retries. If persistent, contact support.

### Results

**Q: What does the attention heatmap show?**  
A: Regions the AI focused on during analysis. Red = high attention.

**Q: What's a good confidence score?**  
A: >90% is high confidence. <70% warrants careful manual review.

**Q: Can I override AI predictions?**  
A: Yes. Your diagnosis is final. Document AI-assisted in report.

**Q: How do I export results?**  
A: Click "Generate Report" → Select format → Download or send to PACS.

### Integration

**Q: Does this work with our PACS?**  
A: Yes. Compatible with all major PACS vendors (Philips, GE, Sectra, etc.).

**Q: Can I access from home?**  
A: Yes, with VPN. Contact IT for remote access setup.

**Q: Does this integrate with Epic/Cerner?**  
A: Yes. Results can be sent to EMR via HL7 FHIR.

---

## Training Resources

### Video Tutorials

1. **Getting Started** (5 min) - First login and basic navigation
2. **Processing Your First Slide** (10 min) - Step-by-step walkthrough
3. **Understanding Results** (15 min) - Interpreting heatmaps and confidence
4. **Clinical Reports** (10 min) - Generating and customizing reports
5. **PACS Integration** (15 min) - Worklist and result delivery

**Access**: `https://histocore.yourhospital.org/training`

### Interactive Demos

**Practice Cases**:
- 20 synthetic cases for training
- Covers common tissue types
- Includes expert annotations
- No patient data

**Sandbox Environment**:
- Test system without affecting production
- Upload your own test slides
- Experiment with settings

### Certification

**Clinical User Certification**:
- Complete 5 video tutorials
- Process 10 practice cases
- Pass 20-question quiz
- Certificate valid 1 year

**Contact**: training@histocore.ai

---

## Appendix

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Process selected slide |
| `Ctrl+R` | Generate report |
| `Ctrl+E` | Export results |
| `Ctrl+Z` | Zoom in on heatmap |
| `Ctrl+X` | Zoom out on heatmap |
| `Esc` | Close modal |

### Supported Tissue Types

- Breast
- Lung
- Colon
- Prostate
- Skin
- Lymph node
- Liver
- Kidney
- Brain
- Thyroid

**Note**: Performance varies by tissue type. Contact clinical support for validation data.

### System Status

**Check System Health**: `https://histocore.yourhospital.org/status`

**Maintenance Windows**: Sundays 2-4 AM (system unavailable)

---

## Contact Information

**Clinical Support**: clinical@histocore.ai | 1-800-HISTOCORE  
**Technical Support**: support@histocore.ai | 24/7  
**Training**: training@histocore.ai  
**Security/Compliance**: compliance@histocore.ai

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-04-27  
**For**: Clinical Users (Pathologists, Clinicians, Technicians)

**Feedback**: We value your input! Email feedback@histocore.ai
