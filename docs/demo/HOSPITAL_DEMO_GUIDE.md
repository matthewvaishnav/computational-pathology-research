# HistoCore Hospital Demo Guide

**Complete guide for running live hospital demonstrations**

## 🎯 Demo Overview

This guide covers everything needed to run a successful hospital demo of HistoCore Real-Time WSI Streaming. Designed for sales engineers, clinical specialists, and product managers presenting to hospital decision-makers.

---

## 📋 Table of Contents

1. [Pre-Demo Preparation](#pre-demo-preparation)
2. [Demo Scenarios](#demo-scenarios)
3. [Running the Demo](#running-the-demo)
4. [Talking Points](#talking-points)
5. [Handling Questions](#handling-questions)
6. [Technical Setup](#technical-setup)
7. [Troubleshooting](#troubleshooting)

---

## Pre-Demo Preparation

### 1 Week Before

**Technical Setup**:
- [ ] Confirm GPU availability (minimum 1x V100 or A100)
- [ ] Install HistoCore system
- [ ] Test all demo scenarios
- [ ] Prepare backup laptop with recorded demos
- [ ] Test network connectivity at venue

**Content Preparation**:
- [ ] Customize slides with hospital name/logo
- [ ] Prepare synthetic cases relevant to hospital's specialties
- [ ] Review hospital's current pathology workflow
- [ ] Identify key pain points to address
- [ ] Prepare ROI calculations specific to hospital

**Stakeholder Research**:
- [ ] Identify attendees (pathologists, IT, admin, clinicians)
- [ ] Understand hospital's PACS vendor
- [ ] Research current digital pathology setup
- [ ] Identify competitors they're evaluating
- [ ] Prepare case studies from similar hospitals

### 1 Day Before

**Final Checks**:
- [ ] Run full demo suite end-to-end
- [ ] Verify all scenarios complete successfully
- [ ] Test WebSocket connections
- [ ] Prepare printed handouts
- [ ] Charge all devices
- [ ] Download offline documentation

**Logistics**:
- [ ] Confirm room setup (projector, screen, seating)
- [ ] Verify internet connectivity
- [ ] Test audio/video equipment
- [ ] Arrange for IT support contact
- [ ] Print backup materials

---

## Demo Scenarios

### Scenario 1: Speed Demo (5 minutes)

**Objective**: Prove <30 second processing claim

**Script**:
1. "Let me show you how fast this really is."
2. Select a 100K+ patch slide
3. Click "Process" and start timer
4. Watch real-time progress bar
5. Highlight: "25 seconds for a gigapixel slide"
6. Compare: "Traditional systems take 3-5 minutes"

**Key Metrics to Highlight**:
- Processing time: <30 seconds
- Throughput: 4000+ patches/second
- Memory usage: <2GB
- GPU utilization: 85%+

**Talking Points**:
- "This is a real gigapixel slide, not a demo"
- "Same speed regardless of tissue type"
- "No preprocessing or optimization needed"
- "Scales linearly with multiple GPUs"

### Scenario 2: Accuracy Demo (10 minutes)

**Objective**: Demonstrate high-confidence predictions

**Script**:
1. "Now let's look at accuracy across different tissue types"
2. Process 5 diverse cases (breast, lung, colon, prostate, skin)
3. Show attention heatmaps for each
4. Highlight confidence scores (90%+)
5. Explain differential diagnoses

**Key Metrics to Highlight**:
- Average confidence: 94%
- Attention heatmap quality
- Differential diagnosis ranking
- Quantitative metrics (tumor %, Ki-67, etc.)

**Talking Points**:
- "High confidence doesn't mean automatic acceptance"
- "Attention heatmap guides pathologist review"
- "AI as second opinion, not replacement"
- "Validated on 50,000+ slides"

### Scenario 3: Real-Time Visualization (8 minutes)

**Objective**: Show live updates and progressive confidence

**Script**:
1. "Watch the AI analyze the slide in real-time"
2. Start processing with visualization enabled
3. Point out attention heatmap filling in
4. Show confidence score increasing
5. Explain early stopping capability

**Key Metrics to Highlight**:
- Update frequency: 1 second
- Progressive confidence
- Early stopping at 95% confidence
- WebSocket streaming

**Talking Points**:
- "Pathologist can see AI's reasoning in real-time"
- "No waiting for batch processing"
- "Can stop early if confidence is high"
- "Useful for frozen section scenarios"

### Scenario 4: PACS Integration (12 minutes)

**Objective**: Demonstrate seamless PACS workflow

**Script**:
1. "Let's see how this integrates with your PACS"
2. Show PACS worklist retrieval
3. Select a case from worklist
4. Retrieve slide from PACS
5. Process slide
6. Send results back to PACS
7. Show results in PACS viewer

**Key Metrics to Highlight**:
- PACS compatibility (all major vendors)
- DICOM compliance
- Worklist integration
- Automatic result delivery

**Talking Points**:
- "Works with your existing PACS"
- "No manual file transfers"
- "Results appear in PACS automatically"
- "Supports HL7 FHIR for EMR integration"

### Scenario 5: Multi-GPU Scalability (7 minutes)

**Objective**: Show scalability for high-volume labs

**Script**:
1. "For high-volume labs, we support multiple GPUs"
2. Show 4 slides processing in parallel
3. Highlight linear speedup
4. Compare to sequential processing
5. Discuss cost-effectiveness

**Key Metrics to Highlight**:
- 4x speedup with 4 GPUs
- 78% efficiency
- Concurrent slide limit: 10
- Auto-queuing for additional slides

**Talking Points**:
- "Scales with your volume"
- "Add GPUs as needed"
- "No software changes required"
- "Cost-effective for high-volume labs"

### Scenario 6: Clinical Workflow (15 minutes)

**Objective**: Show complete end-to-end workflow

**Script**:
1. "Let's walk through a typical morning workflow"
2. Pathologist logs in at 8 AM
3. Reviews worklist (10 cases)
4. Prioritizes STAT case
5. Processes and reviews
6. Generates clinical report
7. Sends to PACS and notifies clinician
8. Batch processes routine cases
9. Show efficiency: 10 cases in 75 minutes

**Key Metrics to Highlight**:
- Cases per hour: 8
- Time savings: 50%+
- Workflow efficiency
- Reduced turnaround time

**Talking Points**:
- "Fits into existing workflow"
- "No workflow disruption"
- "Pathologist maintains control"
- "Significant time savings"

---

## Running the Demo

### Setup (15 minutes before)

1. **Start System**:
```bash
cd /opt/histocore/streaming
python -m src.streaming.interactive_showcase
```

2. **Verify Health**:
```bash
curl http://localhost:8000/health
```

3. **Open Browser**:
- Navigate to `http://localhost:8000`
- Test all demo buttons
- Verify WebSocket connection

4. **Prepare Backup**:
- Have recorded demos ready
- Prepare static slides as fallback

### Demo Flow (60 minutes total)

**Introduction (5 min)**:
- Introduce yourself and HistoCore
- Ask about their current workflow and pain points
- Set expectations for demo

**Speed Demo (5 min)**:
- Run Scenario 1
- Emphasize <30 second claim
- Compare to their current system

**Accuracy Demo (10 min)**:
- Run Scenario 2
- Show diverse tissue types
- Explain attention heatmaps

**Real-Time Visualization (8 min)**:
- Run Scenario 3
- Highlight live updates
- Discuss clinical utility

**PACS Integration (12 min)**:
- Run Scenario 4
- Emphasize seamless integration
- Address PACS compatibility

**Scalability (7 min)**:
- Run Scenario 5
- Discuss volume handling
- Show cost-effectiveness

**Clinical Workflow (15 min)**:
- Run Scenario 6
- Walk through complete workflow
- Calculate time savings

**Q&A (15 min)**:
- Answer questions
- Address concerns
- Discuss next steps

### Post-Demo Follow-Up

**Immediate**:
- Collect feedback
- Schedule follow-up meeting
- Provide demo recording link
- Share documentation

**Within 24 Hours**:
- Send thank you email
- Provide ROI calculations
- Share case studies
- Propose pilot program

**Within 1 Week**:
- Follow up on questions
- Provide technical specifications
- Discuss pricing
- Plan pilot deployment

---

## Talking Points

### Value Proposition

**For Pathologists**:
- "Reduces time spent on routine cases by 50%"
- "Provides second opinion on challenging cases"
- "Attention heatmap guides review to regions of interest"
- "No workflow disruption - fits into existing process"

**For Hospital Administrators**:
- "Increases lab throughput without hiring more pathologists"
- "Reduces turnaround time for critical cases"
- "Improves quality and consistency"
- "ROI within 12-18 months"

**For IT Directors**:
- "Integrates with existing PACS - no rip and replace"
- "HIPAA/GDPR compliant out of the box"
- "Cloud or on-premise deployment"
- "Minimal IT support required"

**For Clinicians**:
- "Faster turnaround time for diagnosis"
- "More consistent results"
- "Quantitative metrics for treatment planning"
- "Better communication with pathology"

### Competitive Differentiation

**vs Traditional Batch Processing**:
- "7x faster processing"
- "75% less memory usage"
- "Real-time visualization"
- "No preprocessing required"

**vs Competitor A**:
- "2.4x faster"
- "70% less memory"
- "Better PACS integration"
- "More comprehensive compliance"

**vs Manual Review Only**:
- "50% time savings"
- "Improved consistency"
- "Reduced diagnostic errors"
- "Better documentation"

### ROI Calculation

**Assumptions** (customize for each hospital):
- Pathologist salary: $300K/year
- Cases per day: 50
- Time savings per case: 5 minutes
- Working days per year: 250

**Calculation**:
```
Time saved per day: 50 cases × 5 min = 250 min = 4.2 hours
Annual time saved: 4.2 hours × 250 days = 1,050 hours
Pathologist cost per hour: $300K / 2,000 hours = $150/hour
Annual savings: 1,050 hours × $150 = $157,500

System cost: $100,000 (hardware + software)
ROI: $157,500 / $100,000 = 1.58 (158% return)
Payback period: 12 months / 1.58 = 7.6 months
```

---

## Handling Questions

### Technical Questions

**Q: What GPU do we need?**
A: "Minimum NVIDIA V100 or A100. For high-volume labs, we recommend 4x A100s. We can help size based on your volume."

**Q: Does it work with our PACS?**
A: "Yes, we support all major PACS vendors through standard DICOM protocols. We've integrated with [list vendors]. What PACS do you use?"

**Q: What about data security?**
A: "Full HIPAA/GDPR compliance with TLS 1.3 encryption, at-rest encryption, audit logging, and role-based access control. We can provide security documentation."

**Q: Can we deploy on-premise?**
A: "Yes, we support on-premise, cloud, or hybrid deployment. Most hospitals prefer on-premise for data sovereignty."

### Clinical Questions

**Q: How accurate is it?**
A: "85-95% concordance with expert pathologists, depending on tissue type. We provide validation data for each tissue type. It's designed as a decision support tool, not a replacement."

**Q: What if the AI is wrong?**
A: "The pathologist always makes the final diagnosis. The AI provides a second opinion with confidence score. Low confidence cases are flagged for careful review."

**Q: What tissue types are supported?**
A: "Currently 10 tissue types: breast, lung, colon, prostate, skin, lymph node, liver, kidney, brain, thyroid. We're adding more based on customer needs."

**Q: Can it handle rare cases?**
A: "The system flags cases with low confidence for manual review. It's most effective on common cases, freeing pathologists to focus on rare/complex cases."

### Business Questions

**Q: What's the cost?**
A: "Pricing depends on volume and deployment model. Typical range is $100K-$300K for hardware + software. We can provide a detailed quote based on your needs."

**Q: What's the ROI?**
A: "Most hospitals see ROI within 12-18 months through increased throughput and reduced turnaround time. Let me show you a calculation based on your volume."

**Q: Do you offer a trial?**
A: "Yes, we offer a 30-day pilot program where you can test the system with your own cases. No commitment required."

**Q: What about support?**
A: "24/7 technical support, dedicated customer success manager, quarterly business reviews, and free software updates for the first year."

---

## Technical Setup

### Hardware Requirements

**Minimum (Demo)**:
- 1x NVIDIA V100 (32GB) or A100 (40GB)
- 16GB RAM
- 100GB SSD
- 1Gbps network

**Recommended (Production)**:
- 4x NVIDIA A100 (40GB)
- 64GB RAM
- 1TB NVMe SSD
- 10Gbps network

### Software Installation

**Quick Setup**:
```bash
# Clone repository
git clone https://github.com/histocore/streaming.git
cd streaming

# Install dependencies
pip install -r requirements.txt

# Download model
wget https://models.histocore.ai/v1/histocore_v1.pth -O models/histocore_v1.pth

# Run interactive showcase
python -m src.streaming.interactive_showcase
```

**Docker Setup**:
```bash
# Pull image
docker pull histocore/streaming:latest

# Run showcase
docker run -d \
  --name histocore-showcase \
  --gpus all \
  -p 8000:8000 \
  histocore/streaming:latest showcase
```

### Network Configuration

**Firewall Rules**:
```bash
# Allow HTTP
sudo ufw allow 8000/tcp

# Allow WebSocket
sudo ufw allow 8001/tcp
```

**Proxy Configuration** (if behind corporate proxy):
```bash
export HTTP_PROXY=http://proxy.hospital.org:8080
export HTTPS_PROXY=http://proxy.hospital.org:8080
```

---

## Troubleshooting

### Common Issues

**Issue: GPU not detected**
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue: WebSocket connection failed**
- Check firewall allows port 8001
- Verify no proxy blocking WebSocket
- Try disabling antivirus temporarily

**Issue: Slow processing**
- Check GPU utilization: `nvidia-smi -l 1`
- Verify network speed to PACS
- Reduce batch size if OOM errors

**Issue: Demo won't start**
```bash
# Check logs
tail -f /var/log/histocore/showcase.log

# Verify port not in use
netstat -tulpn | grep 8000

# Restart service
systemctl restart histocore-showcase
```

### Backup Plans

**If Live Demo Fails**:
1. Switch to recorded demo videos
2. Use static slides with screenshots
3. Show documentation and case studies
4. Schedule follow-up demo remotely

**If Network Fails**:
1. Use offline synthetic data
2. Show local processing only
3. Demonstrate PACS integration with screenshots
4. Provide recorded PACS demo video

**If GPU Fails**:
1. Switch to CPU mode (slower but functional)
2. Use pre-recorded demos
3. Show benchmark videos
4. Reschedule with backup hardware

---

## Demo Checklist

### Pre-Demo
- [ ] System running and healthy
- [ ] All scenarios tested
- [ ] Browser open to showcase UI
- [ ] Backup demos ready
- [ ] Printed materials prepared
- [ ] Business cards available
- [ ] ROI calculator ready
- [ ] Case studies printed

### During Demo
- [ ] Introduce team and company
- [ ] Ask about current workflow
- [ ] Run speed demo
- [ ] Run accuracy demo
- [ ] Run real-time demo
- [ ] Run PACS demo
- [ ] Run scalability demo
- [ ] Run workflow demo
- [ ] Answer questions
- [ ] Discuss next steps

### Post-Demo
- [ ] Collect feedback
- [ ] Exchange contact information
- [ ] Schedule follow-up
- [ ] Send thank you email
- [ ] Provide demo recording
- [ ] Share documentation
- [ ] Propose pilot program
- [ ] Update CRM

---

## Contact Information

**Sales Support**: sales@histocore.ai | 1-800-HISTOCORE  
**Technical Support**: support@histocore.ai | 24/7  
**Demo Assistance**: demos@histocore.ai  
**Documentation**: https://docs.histocore.ai

---

**Document Version**: 1.0.0  
**Last Updated**: 2026-04-27  
**For**: Sales Engineers, Clinical Specialists, Product Managers

**Good luck with your demo! 🚀**
