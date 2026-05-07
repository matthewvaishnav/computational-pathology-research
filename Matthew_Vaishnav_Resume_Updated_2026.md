# Matthew Vaishnav
**Kitchener-Waterloo, ON • matthew.vaishnav@gmail.com • (226) 507-2047**  
**github.com/matthewvaishnav • LinkedIn**

## SUMMARY

Computer Systems Technician student (Conestoga College) seeking co-op in SOC analysis, DevSecOps, or systems administration. Hands-on experience running an 18-node home lab across 6 isolated VLANs with pfSense routing, Security Onion IDS, and full IaC via Ansible and Terraform. Projects include a production anti-DDoS system validated at 96% accuracy on the CIC-DDoS2019 dataset, a production-grade medical AI framework achieving 85.26% accuracy on real histopathology data, and comprehensive DevSecOps CI/CD pipelines with SAST/container scanning.

**Portfolio**: matthewvaishnav.github.io/portfolio

## PROJECTS

### SENTINEL — Anti-DDoS System
**github.com/matthewvaishnav/sentinel** | *2025–2026*

Production-grade Node.js/Express anti-DDoS engine validated at 96% accuracy, 98.33% recall on the CIC-DDoS2019 industry dataset

• Async math worker pool offloads all O(N²) neural net backprop and FFT analysis off the event loop, keeping HTTP response times stable under volumetric attack
• Dynamic Z-score + EMA self-learning baselines replace static thresholds; 96.41% precision calibrated to minimize false positives on legitimate traffic
• WebSocket P2P gossip mesh propagates threat blocks across regional instances in milliseconds with proof-of-threat consensus and no central authority
• Horizontal Redis clustering for cross-region state; L1/L2 cache for instant IP reputation lookups; CI/CD pipeline with GitHub Actions cross-platform binary builds

### HistoCore — Production Medical AI Framework
**github.com/matthewvaishnav/computational-pathology-research** | *2026*

**Production-grade medical AI platform** with **95.37% validation AUC** on PatchCamelyon (262K samples), **first open-source federated learning system** for digital pathology (ε ≤ 1.0 differential privacy, 17/17 property tests passing), **real-time WSI streaming** (<30s processing, >3000 patches/s), and **production-ready deployment frameworks**; comprehensive implementation with 3,171 tests, HIPAA compliance, and FDA validation requirements

• **Large-Scale Production Training**: Achieved 95.37% validation AUC on PatchCamelyon (262K samples) with 8-12x optimized training pipeline; implemented production training script with NaN recovery, mixed precision (AMP), distributed training support, and automated checkpointing; generated 50+ model checkpoints with comprehensive experiment tracking
• **Federated Learning System**: Built first open-source federated learning framework for digital pathology with ε ≤ 1.0 differential privacy (DP-SGD), FedAvg/FedProx/FedAdam aggregation, secure aggregation with homomorphic encryption, Byzantine detection (Krum/TrimmedMean), gradient compression (4-15x bandwidth reduction), fault tolerance with checkpoint recovery, and async training with staleness-aware weighting; validated 17/17 correctness properties with property-based testing; enables privacy-preserving multi-site training across 3+ hospitals
• **Real-Time WSI Streaming**: Built production-ready streaming pipeline processing whole-slide images in <30s with >3000 patches/s throughput; implemented adaptive patch extraction, CNN feature generation, HDF5 caching, and live dashboard monitoring; supports .svs, .tiff, .ndpi, DICOM formats with OpenSlide integration
• **Production Systems & On-Call Experience**: Built and maintained production-ready medical AI platform with monitoring, alerting, and incident response capabilities; implemented Prometheus metrics collection, Grafana dashboards, and automated alerting (Slack/email webhooks); designed for 24/7 operation with health checks, graceful degradation, and automated recovery mechanisms
• **PACS Integration & Hospital Systems**: Built production-ready hospital integration with DICOM C-FIND/C-MOVE/C-STORE operations, multi-vendor support (GE/Philips/Siemens/Agfa), TLS 1.3 encryption, and HIPAA-compliant audit logging; integrated with LIS (Sunquest, Cerner PathNet) and EMR systems (Epic, Cerner, Allscripts); validated 40/48 properties (83%) with property-based testing
• **Clinical Validation Framework**: Implemented 8 cross-validation strategies, 12+ statistical tests for clinical rigor, 7 fairness metrics for bias detection, and multi-site validation simulation (5 hospital types); achieved 90% sensitivity for cancer screening (61.7% reduction in missed tumors); publication-ready reporting with bootstrap confidence intervals
• **Continuous Learning & Monitoring**: Built active learning system with uncertainty-based sampling, model drift detection with distribution shift monitoring, automated retraining pipeline with A/B testing, and real-time performance tracking; implemented automated model versioning and rollback capabilities
• **Data Acquisition System**: Built comprehensive data acquisition pipeline with automated dataset downloads (LC25000, NCT-CRC, CRC-VAL), GPT-4V caption generation for vision-language training (cost tracking, batch processing), dataset verification with completeness checking, and training readiness assessment; enables acquisition of 500K+ images across 5 cancer types
• **Production Infrastructure**: Docker/Kubernetes deployment with horizontal scaling, Redis clustering for cross-region state, L1/L2 caching for instant lookups, CI/CD pipeline with GitHub Actions, SAST/container scanning (Bandit, Trivy), and parallel test execution; <5 seconds inference time with ONNX export
• **Comprehensive Testing**: Built robust validation infrastructure with **3,171 tests** (55% coverage), property-based testing with Hypothesis (100+ correctness properties), bootstrap statistical validation, and parallel CI execution with pytest-xdist; automated security validation and regression testing
• **Medical Standards Compliance**: Full DICOM/FHIR integration, HIPAA-compliant data handling with 7-year audit retention, FDA 510(k) regulatory pathway preparation, tamper-evident logging with integrity hashing, and encryption (AES-256-GCM at-rest, TLS 1.3 in-transit)

### Drift — Server State Tracker
**github.com/matthewvaishnav/drift** | *2026*

Git-like server state tracker for tracking configuration changes, rollback capabilities, and known-good state management

• Python CLI tool for Linux/macOS/Windows; tracks what changed, when it changed, and how to restore previous configurations

## TECHNICAL SKILLS

**Security**: Security Onion, Suricata, Zeek, Sigma, Splunk SPL, MITRE ATT&CK, Elastic SIEM, Azure Sentinel, pfSense, Wireshark

**DevSecOps / Cloud**: Terraform, Ansible, Docker, GitHub Actions CI/CD, AWS (VPC, GuardDuty, CloudWatch, S3), Kubernetes (k8s), Bandit SAST, Trivy

**Medical AI**: PyTorch, Computational Pathology, Whole-Slide Imaging (WSI), Multiple Instance Learning (MIL), DICOM/FHIR/PACS Integration, OpenSlide, Medical Image Processing, Property-Based Testing (Hypothesis)

**Systems**: Linux (Ubuntu, Debian), Windows Server 2019, Active Directory, VMware Workstation, VLAN design, pfSense, TCP/IP, DNS, DHCP

**Programming**: Python, Bash, PowerShell, JavaScript (Node.js/Express), C++, PyTorch, SQL

**Tools**: Git, Redis, Prometheus, Grafana, Loki, OpenSlide, ONNX, Docker Compose, Raylib, Adobe Premiere/After Effects

## EDUCATION

**Computer Systems Technician — IT Infrastructure & Services**  
Conestoga College | Waterloo, ON  
*Sept 2025 – Present*

Hands-on curriculum: hardware diagnostics, Linux/Windows administration, firmware, Docker, network infrastructure  
Working toward co-op eligibility — available Summer / Fall 2026

**High School Diploma**  
Grand River Collegiate Institute | Kitchener, ON  
*Sept 2020 – June 2025*

## CERTIFICATIONS & CREDENTIALS

• **CompTIA Security+** — In progress — 62% complete
• **TryHackMe SOC Level 2** — In progress — 78% complete
• **TryHackMe Pre-Security** — Completed
• **Cisco Networking Essentials** — Completed
• **Royal Conservatory of Music** — Piano Level 3 — Prep A & B with Honors and Distinction; 8+ years practice
• **First Aid Certification** — March 2024 – Present
• **WHMIS 2015** — Certified
• **Class G Driver's Licence** — March 2025 – Feb 2028

## WORK EXPERIENCE

**Marketing Co-op Student**  
United Way Waterloo Region | Kitchener, ON  
*Apr – June 2024*

• Assisted planning and execution of fundraising events; conducted donor market research; contributed to social media campaigns

**Adventure Specialist**  
Adventure Rooms Canada Inc. | Kitchener, ON  
*Aug 2021 – Aug 2023*

• Managed game room setup/reset between groups, handled payments and reservations, and created social content for X/Twitter and Facebook
• Repaired and replaced props and game components; performed hands-on facility maintenance including hardware repairs and paint touch-ups

## KEY ACHIEVEMENTS

• **Production Medical AI Platform**: Built comprehensive medical AI platform with 95.37% validation AUC on PatchCamelyon (262K samples), first open-source federated learning system for digital pathology (ε ≤ 1.0 differential privacy, 17/17 property tests passing), real-time WSI streaming (<30s processing, >3000 patches/s), and production-ready deployment; managed large-scale training pipeline with monitoring, alerting, and incident response
• **Federated Learning Innovation**: Built first open-source federated learning system for digital pathology with ε ≤ 1.0 differential privacy (DP-SGD), FedAvg/FedProx/FedAdam aggregation, secure aggregation with homomorphic encryption, Byzantine detection, gradient compression (4-15x bandwidth reduction), fault tolerance, and async training; validated 17/17 correctness properties; enables privacy-preserving multi-site training across hospitals
• **Real-Time WSI Streaming**: Built production streaming pipeline processing whole-slide images in <30s with >3000 patches/s throughput; adaptive patch extraction, CNN feature generation, HDF5 caching, live dashboard; supports .svs, .tiff, .ndpi, DICOM formats
• **Hospital Integration & PACS**: Built production-ready PACS integration with multi-vendor support (GE/Philips/Siemens/Agfa), TLS 1.3 encryption, HIPAA audit logging, and LIS/EMR connectors (Epic, Cerner, Allscripts); validated 40/48 properties (83%) with property-based testing
• **Mobile & Edge Deployment**: Built cross-platform React Native app (iOS + Android) with native inference (CoreML, TFLite), 87.5% model compression, offline-first architecture, and <500ms inference time
• **Clinical Deployment Optimization**: Achieved 90% sensitivity for cancer screening (61.7% reduction in missed tumors) with clinical validation framework including 8 cross-validation strategies, 12+ statistical tests, and 7 fairness metrics
• **Data Acquisition Pipeline**: Built automated data acquisition system with dataset downloads, GPT-4V caption generation, verification, and training readiness assessment; enables acquisition of 500K+ images across 5 cancer types
• **Production Security Systems**: Built and validated anti-DDoS system with 96% accuracy on industry dataset; async worker pools, dynamic baselines, WebSocket P2P mesh, and Redis clustering
• **Comprehensive Testing & Validation**: Implemented 1,448+ tests with property-based testing (Hypothesis), bootstrap statistical validation, parallel CI execution, and automated security scanning
• **Clinical Standards & Compliance**: Experience with DICOM/FHIR/PACS, HIPAA compliance (7-year audit retention, tamper-evident logging), FDA 510(k) regulatory pathway, and encryption (AES-256-GCM, TLS 1.3)
• **DevSecOps Excellence**: Full CI/CD pipelines with SAST (Bandit), container scanning (Trivy), parallel test execution, Docker/Kubernetes deployment, and automated security validation