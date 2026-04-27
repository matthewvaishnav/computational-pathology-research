# HistoCore Real-Time WSI Streaming - Video Tutorial Scripts

**Production-Ready Scripts for Training Videos**

## 📹 Tutorial Series Overview

5 video tutorials covering clinical user workflows:

1. **Getting Started** (5 min) - First login and navigation
2. **Processing Your First Slide** (10 min) - Complete walkthrough
3. **Understanding Results** (15 min) - Interpreting outputs
4. **Clinical Reports** (10 min) - Report generation
5. **PACS Integration** (15 min) - Worklist workflows

---

## Tutorial 1: Getting Started (5 minutes)

### Learning Objectives
- Access the system
- Navigate the dashboard
- Understand user interface layout
- Check system status

### Script

**[INTRO - 0:00-0:30]**

"Welcome to HistoCore Real-Time WSI Streaming. I'm Dr. Sarah Chen, and in this 5-minute tutorial, you'll learn how to access the system and navigate the dashboard. By the end, you'll be ready to process your first slide."

**[SCREEN: Login page]**

**[SECTION 1: Logging In - 0:30-1:30]**

"Let's start by logging in. Open your browser and navigate to histocore.yourhospital.org."

**[ACTION: Type URL in browser]**

"You'll see the login page. Use your hospital credentials - the same ones you use for Epic or Cerner."

**[ACTION: Enter username and password]**

"If your hospital uses Single Sign-On, you'll be redirected to your identity provider. Just follow the prompts."

**[ACTION: Click Login]**

"And we're in. That's it - no special credentials needed."

**[SECTION 2: Dashboard Overview - 1:30-3:00]**

**[SCREEN: Main dashboard]**

"This is your main dashboard. Let's walk through the key areas."

**[ACTION: Highlight top navigation]**

"At the top, you have your main navigation: Dashboard, Process Slide, Worklist, Results, and Reports."

**[ACTION: Highlight left sidebar]**

"On the left, you'll see recent cases and quick actions. This is where you'll spend most of your time."

**[ACTION: Highlight center panel]**

"The center shows your active processing queue. Right now it's empty because we haven't processed anything yet."

**[ACTION: Highlight right panel]**

"On the right, you have system status - GPU availability, processing capacity, and any alerts."

**[SECTION 3: System Status - 3:00-4:00]**

**[ACTION: Click system status icon]**

"Let's check system status. Click the status icon in the top right."

**[SCREEN: System status modal]**

"Green means everything is operational. You'll see:
- GPU availability - how many GPUs are free
- Processing capacity - how many slides can be processed concurrently
- Queue length - how many slides are waiting

If you see yellow or red, contact IT before processing critical cases."

**[ACTION: Close modal]**

**[SECTION 4: User Profile - 4:00-4:30]**

**[ACTION: Click user profile icon]**

"Click your profile icon to see your role and permissions."

**[SCREEN: User profile]**

"Your role determines what you can do:
- Pathologists can process, review, and approve
- Clinicians can process and view
- Technicians can process
- Viewers can only view results

You can also change your password and notification preferences here."

**[OUTRO - 4:30-5:00]**

"That's it for getting started. You now know how to log in, navigate the dashboard, and check system status. In the next tutorial, we'll process your first slide. See you there!"

**[END SCREEN: Next Tutorial Preview]**

---

## Tutorial 2: Processing Your First Slide (10 minutes)

### Learning Objectives
- Upload a slide
- Start processing
- Monitor real-time progress
- View initial results

### Script

**[INTRO - 0:00-0:30]**

"Welcome back. In this tutorial, you'll process your first whole slide image and watch the AI analyze it in real-time. This takes about 30 seconds for a typical slide."

**[SCREEN: Dashboard]**

**[SECTION 1: Uploading a Slide - 0:30-2:30]**

"Let's start by uploading a slide. Click 'Process Slide' in the top navigation."

**[ACTION: Click Process Slide]**

**[SCREEN: Upload interface]**

"You have two options: upload a file directly, or retrieve from PACS. We'll upload directly first."

**[ACTION: Click 'Upload File']**

"Click 'Choose File' and select your slide. We support .svs, .tiff, .ndpi, and DICOM formats."

**[ACTION: Select file from computer]**

"I'm selecting a breast biopsy slide - about 100,000 patches, typical size."

**[SCREEN: File selected, metadata form]**

"You can optionally enter metadata like patient ID, case number, or diagnosis. This helps with organization but isn't required."

**[ACTION: Fill in optional fields]**

"For this demo, I'll add a case number: BC-2026-001."

**[ACTION: Click 'Upload & Process']**

"Click 'Upload & Process'. The system will upload the file and start processing immediately."

**[SECTION 2: Real-Time Processing - 2:30-6:00]**

**[SCREEN: Processing interface with progress bar]**

"Now we're in the processing view. Let's look at what's happening."

**[ACTION: Highlight progress bar]**

"The progress bar shows completion percentage. We're at 15% after just 5 seconds."

**[ACTION: Highlight attention heatmap]**

"This is the attention heatmap - it updates in real-time as the AI processes tiles. Red areas are where the AI is focusing attention."

**[WAIT: Let processing continue to ~50%]**

"Notice how the heatmap is filling in. The AI is scanning the slide systematically, identifying regions of interest."

**[ACTION: Highlight confidence score]**

"The confidence score is increasing as more of the slide is processed. We're at 72% confidence now."

**[ACTION: Highlight processing stats]**

"Down here you see processing statistics:
- Patches processed: 52,000 of 100,000
- Processing speed: 3,500 patches per second
- Estimated time remaining: 12 seconds

This is running on a single GPU. With multiple GPUs, it's even faster."

**[WAIT: Let processing complete]**

**[SCREEN: Processing complete notification]**

"And we're done! 28 seconds total. The system automatically saves the results."

**[SECTION 3: Initial Results Review - 6:00-9:00]**

**[SCREEN: Results view]**

"Let's look at the results. The attention heatmap is now complete."

**[ACTION: Zoom in on high-attention region]**

"I can zoom in on high-attention areas. See these red regions? The AI identified these as diagnostically significant."

**[ACTION: Zoom out, show full slide]**

"The confidence score is 94% - that's high confidence. But remember, this is a decision support tool, not a replacement for your expertise."

**[ACTION: Highlight prediction]**

"The prediction is 'Invasive Ductal Carcinoma' with 94% confidence. The system also shows top 3 differential diagnoses."

**[ACTION: Show quantitative metrics]**

"Below, you see quantitative metrics:
- Tumor percentage: 35%
- Mitotic count: 12 per 10 HPF
- Ki-67 index: 28%

These are estimates - always verify with your own assessment."

**[SECTION 4: Next Steps - 9:00-9:30]**

"From here, you can:
- Generate a clinical report (we'll cover this in Tutorial 4)
- Export results to PACS
- Download the attention heatmap
- Process another slide"

**[OUTRO - 9:30-10:00]**

"Congratulations! You've processed your first slide. In the next tutorial, we'll dive deeper into understanding and interpreting results. See you there!"

**[END SCREEN: Next Tutorial Preview]**

---

## Tutorial 3: Understanding Results (15 minutes)

### Learning Objectives
- Interpret attention heatmaps
- Understand confidence scores
- Evaluate prediction quality
- Identify when to trust AI vs manual review

### Script

**[INTRO - 0:00-0:30]**

"Welcome to Tutorial 3. Now that you can process slides, let's learn how to interpret the results. We'll cover attention heatmaps, confidence scores, and when to trust the AI."

**[SCREEN: Results view with completed analysis]**

**[SECTION 1: Attention Heatmaps Deep Dive - 0:30-5:00]**

"Let's start with the attention heatmap. This is the most important visualization."

**[ACTION: Show full slide with heatmap overlay]**

"The heatmap shows where the AI focused during analysis. Think of it as the AI's 'gaze' across the slide."

**[ACTION: Highlight color scale]**

"The color scale goes from blue (low attention) to red (high attention). Blue areas are typically normal tissue or background. Red areas are where the AI found something interesting."

**[ACTION: Zoom to high-attention region]**

"Let's zoom into this red region. What makes this area interesting to the AI?"

**[SCREEN: Zoomed view showing cellular details]**

"You can see increased cellular density, nuclear atypia, and mitotic figures. The AI correctly identified this as a region of concern."

**[ACTION: Zoom to another high-attention region]**

"Here's another red region. Notice the different morphology - this is a different tumor focus."

**[ACTION: Show multiple red regions]**

"Multiple red regions indicate heterogeneous features. This is common in complex cases."

**[ACTION: Zoom to blue region]**

"Now let's look at a blue region. This is normal stroma - the AI correctly identified it as less diagnostically significant."

**[KEY POINT]**

"Important: The heatmap guides your review, but doesn't replace it. Always examine high-attention areas yourself."

**[SECTION 2: Confidence Scores - 5:00-8:00]**

**[SCREEN: Confidence score display]**

"Now let's talk about confidence scores. This slide has 94% confidence."

**[ACTION: Show confidence scale]**

"We categorize confidence into three ranges:
- 90-100%: High confidence - AI is very certain
- 70-89%: Moderate confidence - review carefully
- Below 70%: Low confidence - manual review strongly recommended"

**[ACTION: Show example of high confidence case]**

"High confidence cases like this one typically have:
- Clear diagnostic features
- Good slide quality
- Sufficient diagnostic tissue
- Minimal artifacts"

**[ACTION: Show example of low confidence case]**

"Low confidence cases might have:
- Ambiguous features
- Poor slide quality (out of focus, staining issues)
- Limited diagnostic tissue
- Significant artifacts"

**[KEY POINT]**

"Confidence score ≠ accuracy. A high confidence score means the AI is certain, but it could still be wrong. Always review the attention heatmap and use your clinical judgment."

**[SECTION 3: Prediction Quality Assessment - 8:00-11:00]**

**[SCREEN: Prediction results]**

"Let's evaluate prediction quality. The AI predicted 'Invasive Ductal Carcinoma' with 94% confidence."

**[ACTION: Show top 3 differential diagnoses]**

"The system also shows differential diagnoses:
1. Invasive Ductal Carcinoma - 94%
2. Invasive Lobular Carcinoma - 4%
3. Ductal Carcinoma In Situ - 2%

This distribution suggests the AI is very confident in IDC."

**[ACTION: Show quantitative metrics]**

"Quantitative metrics provide additional context:
- Tumor percentage: 35%
- Mitotic count: 12 per 10 HPF
- Ki-67 index: 28%"

**[ACTION: Compare with attention heatmap]**

"Cross-reference these metrics with the heatmap. Does the tumor percentage match the red regions? Do high-attention areas show mitotic activity?"

**[CASE STUDY: Show discordant case]**

"Here's an example where the AI got it wrong. The prediction was 'Benign' with 88% confidence, but this is actually a low-grade carcinoma."

**[ACTION: Show attention heatmap of discordant case]**

"Notice the heatmap is mostly blue - the AI missed the subtle features. This is why you always review."

**[SECTION 4: When to Trust AI vs Manual Review - 11:00-14:00]**

"So when should you trust the AI, and when should you do a full manual review?"

**[SCREEN: Decision flowchart]**

"Here's a practical framework:

**Trust AI as a strong second opinion when:**
- Confidence > 90%
- Attention heatmap makes clinical sense
- Quantitative metrics are consistent
- Slide quality is good
- Case is straightforward

**Always do full manual review when:**
- Confidence < 70%
- Attention heatmap doesn't match your assessment
- Discordant clinical context
- Poor slide quality
- Complex or rare case
- Medicolegal implications"

**[CASE EXAMPLES]**

"Let's look at examples:

**Example 1: Trust AI**
- Clear IDC, 96% confidence
- Heatmap highlights obvious tumor
- Metrics consistent
- Good slide quality
→ Use AI as confirmation, brief review

**Example 2: Full Manual Review**
- Atypical features, 68% confidence
- Heatmap shows scattered attention
- Metrics inconsistent
- Borderline case
→ Full manual review required"

**[KEY POINT]**

"The AI is a tool, not a decision-maker. Use it to enhance your workflow, not replace your expertise."

**[OUTRO - 14:00-15:00]**

"You now understand how to interpret attention heatmaps, confidence scores, and when to trust AI predictions. In the next tutorial, we'll generate clinical reports. See you there!"

**[END SCREEN: Next Tutorial Preview]**

---

## Tutorial 4: Clinical Reports (10 minutes)

### Learning Objectives
- Generate PDF reports
- Customize report templates
- Export to PACS/EMR
- Add pathologist annotations

### Script

**[INTRO - 0:00-0:30]**

"Welcome to Tutorial 4. In this tutorial, you'll learn how to generate professional clinical reports with AI results, customize templates, and export to your PACS or EMR."

**[SCREEN: Results view]**

**[SECTION 1: Generating Basic Report - 0:30-3:00]**

"Let's start by generating a basic report. From the results view, click 'Generate Report'."

**[ACTION: Click Generate Report button]**

**[SCREEN: Report template selection]**

"You'll see three template options:

1. **Standard Diagnostic Report** - Full patient info, clinical context, AI findings, pathologist interpretation
2. **Research Report** - De-identified, detailed metrics, no PHI
3. **Teaching Case Report** - Educational annotations, differential diagnosis discussion"

**[ACTION: Select Standard Diagnostic Report]**

"For clinical use, select 'Standard Diagnostic Report'."

**[SCREEN: Report customization interface]**

"Now you can customize the report before generating."

**[ACTION: Review pre-filled sections]**

"The system pre-fills:
- Patient demographics (from PACS)
- Slide metadata (size, magnification, staining)
- AI prediction and confidence
- Attention heatmap visualization
- Processing statistics"

**[ACTION: Click Generate PDF]**

"Click 'Generate PDF'. This takes about 5 seconds."

**[SCREEN: Generated PDF preview]**

"Here's your report. Let's review the sections."

**[SECTION 2: Report Sections - 3:00-6:00]**

**[ACTION: Scroll through PDF]**

"**Section 1: Patient Information**
- Name, MRN, DOB
- Accession number
- Clinical indication"

**[ACTION: Scroll to next section]**

"**Section 2: Specimen Information**
- Specimen type and site
- Slide ID and staining
- Scan date and magnification"

**[ACTION: Scroll to AI findings]**

"**Section 3: AI-Assisted Analysis**
- Primary diagnosis with confidence
- Differential diagnoses
- Quantitative metrics
- Attention heatmap visualization"

**[ACTION: Highlight heatmap in report]**

"The attention heatmap is embedded in the report. This helps clinicians understand what the AI focused on."

**[ACTION: Scroll to pathologist section]**

"**Section 4: Pathologist Interpretation**
- This is where you add your assessment
- You can agree, disagree, or add nuance
- Required for final sign-off"

**[ACTION: Scroll to footer]**

"**Section 5: Metadata**
- Processing timestamp
- AI model version
- Pathologist name and signature
- Audit trail"

**[SECTION 3: Customizing Reports - 6:00-8:00]**

"Let's customize a report. Click 'Customize' before generating."

**[ACTION: Click Customize button]**

**[SCREEN: Customization interface]**

"You can:
- Add clinical notes
- Include/exclude sections
- Add institutional logo
- Change language (English, Spanish, French, etc.)"

**[ACTION: Add clinical notes]**

"I'll add a clinical note: 'Patient has family history of breast cancer. Correlate with imaging findings.'"

**[ACTION: Upload institutional logo]**

"Upload your hospital logo for branding."

**[ACTION: Select sections to include]**

"You can exclude sections. For example, uncheck 'Processing Statistics' if you don't want technical details in the clinical report."

**[ACTION: Click Generate PDF]**

"Generate the customized report."

**[SCREEN: Customized PDF]**

"Notice the clinical note is included, and the hospital logo appears in the header."

**[SECTION 4: Exporting and Delivery - 8:00-9:30]**

"Now let's export the report."

**[ACTION: Click Export button]**

**[SCREEN: Export options]**

"You have several options:

1. **Download PDF** - Save to your computer
2. **Send to PACS** - Automatically uploads to PACS as DICOM SR
3. **Export to EMR** - Sends to Epic/Cerner via HL7 FHIR
4. **Email to Clinician** - Direct email delivery"

**[ACTION: Select Send to PACS]**

"For clinical workflow, 'Send to PACS' is most common. This creates a DICOM Structured Report and uploads it to your PACS."

**[ACTION: Click Send]**

"Click 'Send'. The system confirms successful upload."

**[SCREEN: Success notification]**

"The report is now in PACS, linked to the original study. Clinicians can view it alongside the slide."

**[OUTRO - 9:30-10:00]**

"You now know how to generate, customize, and export clinical reports. In the final tutorial, we'll cover PACS integration and worklist workflows. See you there!"

**[END SCREEN: Next Tutorial Preview]**

---

## Tutorial 5: PACS Integration (15 minutes)

### Learning Objectives
- Access PACS worklist
- Retrieve slides from PACS
- Process cases from worklist
- Send results back to PACS

### Script

**[INTRO - 0:00-0:30]**

"Welcome to the final tutorial. In this 15-minute session, you'll learn how to integrate HistoCore with your PACS system. We'll cover worklist access, slide retrieval, and result delivery."

**[SCREEN: Dashboard]**

**[SECTION 1: Accessing PACS Worklist - 0:30-3:00]**

"Let's start by accessing the PACS worklist. Click 'Worklist' in the top navigation."

**[ACTION: Click Worklist]**

**[SCREEN: PACS worklist interface]**

"This is your PACS worklist. It shows all pending digital pathology cases from your PACS system."

**[ACTION: Highlight worklist columns]**

"The worklist displays:
- Patient name and MRN
- Accession number
- Study date
- Modality (WSI - Whole Slide Imaging)
- Number of slides
- Status (Pending, In Progress, Complete)"

**[ACTION: Show filter options]**

"You can filter by:
- Date range
- Patient name or MRN
- Accession number
- Status
- Priority (Routine, Urgent, STAT)"

**[ACTION: Apply filter for today's cases]**

"Let's filter for today's cases. I'll select 'Today' from the date range."

**[SCREEN: Filtered worklist]**

"Now we see only today's pending cases - 12 cases total."

**[SECTION 2: Retrieving Slides from PACS - 3:00-6:00]**

"Let's retrieve a slide from PACS. Click on a case to open it."

**[ACTION: Click on case]**

**[SCREEN: Case details]**

"Here are the case details:
- Patient: Jane Doe, MRN 123456
- Accession: BC-2026-042
- Study date: Today
- Slides: 3 slides available"

**[ACTION: Show slide thumbnails]**

"The system shows thumbnails of available slides. I can see:
- Slide 1: H&E, 40x magnification
- Slide 2: H&E, 20x magnification
- Slide 3: IHC (ER), 40x magnification"

**[ACTION: Select slide 1]**

"I'll select Slide 1 for processing."

**[ACTION: Click 'Retrieve & Process']**

"Click 'Retrieve & Process'. The system will:
1. Query PACS for the slide
2. Download the slide data
3. Start processing automatically"

**[SCREEN: Retrieval progress]**

"You'll see retrieval progress. For large slides, this can take 30-60 seconds depending on network speed."

**[SCREEN: Processing starts]**

"Once retrieved, processing starts immediately. We're now in the familiar processing view from Tutorial 2."

**[SECTION 3: Batch Processing from Worklist - 6:00-9:00]**

"You can also process multiple cases at once. Let's go back to the worklist."

**[ACTION: Return to worklist]**

**[SCREEN: Worklist with multiple cases]**

"Select multiple cases by checking the boxes."

**[ACTION: Select 5 cases]**

"I've selected 5 cases - all routine breast biopsies."

**[ACTION: Click 'Batch Process']**

"Click 'Batch Process' at the top."

**[SCREEN: Batch processing confirmation]**

"The system confirms:
- 5 cases selected
- 8 slides total (some cases have multiple slides)
- Estimated time: 4 minutes (with 4 GPUs)"

**[ACTION: Click 'Start Batch Processing']**

"Click 'Start Batch Processing'."

**[SCREEN: Batch processing queue]**

"Now you see the batch processing queue. Cases are processed in parallel based on available GPUs."

**[ACTION: Highlight queue status]**

"The queue shows:
- Case 1: Processing (45% complete)
- Case 2: Processing (30% complete)
- Case 3: Processing (15% complete)
- Case 4: Queued
- Case 5: Queued"

**[ACTION: Wait for first case to complete]**

"Case 1 just completed. The system automatically moves to the next queued case."

**[KEY POINT]**

"You can process up to 10 slides concurrently. Additional slides are automatically queued."

**[SECTION 4: Sending Results to PACS - 9:00-12:00]**

"Once processing completes, let's send results back to PACS."

**[ACTION: Open completed case]**

**[SCREEN: Results view]**

"From the results view, click 'Send to PACS'."

**[ACTION: Click Send to PACS]**

**[SCREEN: PACS delivery options]**

"You have options for what to send:

1. **Attention Heatmap** - Overlay image
2. **Structured Report** - DICOM SR with findings
3. **Both** (recommended)"

**[ACTION: Select Both]**

"Select 'Both' to send the heatmap and structured report."

**[ACTION: Click Send]**

"Click 'Send'. The system creates DICOM objects and uploads to PACS."

**[SCREEN: Upload progress]**

"Upload progress is shown. This usually takes 10-15 seconds."

**[SCREEN: Success notification]**

"Success! The results are now in PACS."

**[ACTION: Show PACS viewer (if available)]**

"If we open the case in the PACS viewer, we can see:
- Original slide
- Attention heatmap overlay
- Structured report with AI findings"

**[SECTION 5: Worklist Management - 12:00-14:00]**

"Let's cover some worklist management features."

**[ACTION: Return to worklist]**

**[SCREEN: Worklist]**

"**Prioritization**

You can change case priority. Right-click on a case and select 'Set Priority'."

**[ACTION: Right-click, set priority to STAT]**

"STAT cases are processed first, even if other cases are queued."

**[ACTION: Show STAT case moved to top]**

"Notice the STAT case moved to the top of the queue."

**[ACTION: Show status updates]**

"**Status Tracking**

Case status updates automatically:
- Pending → In Progress → Complete
- You can also manually mark cases as 'On Hold' or 'Cancelled'"

**[ACTION: Show completed cases]**

"**Completed Cases**

Click 'Completed' to see finished cases. You can:
- Review results
- Regenerate reports
- Re-send to PACS if needed"

**[OUTRO - 14:00-15:00]**

"Congratulations! You've completed all 5 tutorials. You now know how to:
1. Access and navigate the system
2. Process slides
3. Interpret results
4. Generate clinical reports
5. Integrate with PACS

You're ready to use HistoCore in your clinical workflow. For additional help, refer to the Clinical User Guide or contact support at support@histocore.ai."

**[END SCREEN: Certification information]**

"Complete the certification quiz to receive your Clinical User Certificate. Good luck!"

---

## Production Notes

### Video Production Requirements

**Equipment**:
- Screen recording software (Camtasia, OBS Studio)
- High-quality microphone
- 1920x1080 resolution minimum
- 60 FPS for smooth animations

**Editing**:
- Add captions/subtitles
- Include chapter markers
- Add zoom effects for important UI elements
- Background music (optional, low volume)

**Hosting**:
- Upload to hospital LMS
- Also available on YouTube (unlisted)
- Embed in training portal

### Interactive Elements

**Quizzes** (after each tutorial):
- 5 multiple-choice questions
- Must score 80% to proceed
- Immediate feedback

**Practice Environment**:
- Sandbox with synthetic data
- Users can practice without affecting production
- Reset button to start over

### Certification

**Requirements**:
- Complete all 5 tutorials
- Pass all quizzes (80%+)
- Process 10 practice cases
- Pass final exam (20 questions, 85%+)

**Certificate**:
- Digital certificate with unique ID
- Valid for 1 year
- Required for production access

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-04-27  
**For**: Training Video Production Team
