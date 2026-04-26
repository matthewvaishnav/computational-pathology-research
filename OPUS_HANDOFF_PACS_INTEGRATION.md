    # PACS Integration System - Opus 4.7 Handoff

## Context
HistoCore = production-grade computational pathology framework. Need PACS integration for clinical deployment in hospitals.

## What's Done (30%)
- Tasks 1-4.3 complete
- Core adapters: QueryEngine, RetrievalEngine, StorageEngine, SecurityManager, ConfigurationManager
- Foundation laid, NOT production-ready

## What You Must Build (70%)
**Location**: `look here ai agents for specs/specs/pacs-integration-system/`
- `requirements.md` - 12 requirements (PACS query, retrieval, storage, multi-vendor, security, config, error handling, workflow, performance, DICOM parsing, notifications, audit)
- `design.md` - Technical architecture
- `tasks.md` - Implementation plan (Tasks 5-15)

## Critical Tasks (Priority Order)

### 1. Multi-Vendor PACS Support (Task 5)
**Why Critical**: Hospitals use GE/Philips/Siemens/Agfa PACS. Must work with all.
**Files**: `src/clinical/pacs/vendor_adapters.py`
**What**: 
- Vendor-specific DICOM tag handling
- Conformance negotiation
- Vendor optimizations
**Complexity**: High - medical domain knowledge, DICOM protocol quirks per vendor

### 2. Error Handling & Recovery (Task 6)
**Why Critical**: Hospital networks unreliable. Must handle failures gracefully.
**Files**: `src/clinical/pacs/error_handling.py`, `src/clinical/pacs/failover.py`
**What**:
- NetworkErrorHandler with exponential backoff
- DicomErrorHandler for protocol errors
- DeadLetterQueue for failed operations
- FailoverManager for multi-endpoint redundancy
- Circuit breaker patterns
**Complexity**: High - production reliability, edge cases

### 3. Workflow Orchestration (Task 8)
**Why Critical**: Automates query→retrieve→analyze→store pipeline. Core value prop.
**Files**: `src/clinical/pacs/workflow_orchestrator.py`
**What**:
- Automated polling for new WSI studies
- Priority-based processing queues
- Workflow sequencing (query→retrieve→analyze→store)
- Integration with existing ClinicalWorkflowSystem
- Performance optimization (connection pooling, throttling)
**Complexity**: Very High - coordinates all components, performance-critical

### 4. Clinical Notification System (Task 10)
**Why Critical**: Pathologists need alerts when AI results ready.
**Files**: `src/clinical/pacs/notification_system.py`
**What**:
- Multi-channel: email, SMS, HL7 messages
- Critical finding escalation
- Delivery tracking & retry
- Hospital communication system integration
**Complexity**: Medium - integration with external systems

### 5. Audit & Compliance Logging (Task 11)
**Why Critical**: HIPAA compliance mandatory for clinical deployment.
**Files**: `src/clinical/pacs/audit_logger.py`
**What**:
- DICOM operation logging (timestamps, user IDs, operation details)
- HIPAA-compliant audit message format
- PHI access detail logging
- Tamper-evident storage (cryptographic signatures)
- Configurable retention (1-10 years)
- Automatic archiving
- Search & reporting for compliance audits
**Complexity**: High - regulatory compliance, security

### 6. DICOM Parsing Enhancement (Task 9)
**Why Critical**: Robust DICOM handling for WSI metadata & SR generation.
**Files**: `src/clinical/dicom_adapter.py` (extend existing)
**What**:
- Enhanced DICOMMetadata class
- Explicit/implicit VR handling
- Compression codec support (JPEG 2000, JPEG-LS)
- TID 1500 (Measurement Report) template
- AI algorithm identification sequences
**Complexity**: High - DICOM standard complexity

### 7. Integration & System Wiring (Task 13)
**Why Critical**: Connects all components into working system.
**Files**: `src/clinical/pacs/pacs_service.py`
**What**:
- Wire PACS components with ClinicalWorkflowSystem
- Integrate with WSI processing pipeline
- Main service class (startup, shutdown, health checks)
- Configuration loading & validation
- End-to-end integration tests
**Complexity**: Very High - system integration, testing

## Why Opus 4.7 (Not Sonnet 4.5)

### Sonnet 4.5 Limitations
- Struggles with deep medical domain knowledge (DICOM, HL7 FHIR, PACS protocols)
- Multi-vendor integration complexity (vendor-specific quirks)
- Regulatory compliance (HIPAA, FDA/CE marking)
- Security-critical implementation (TLS 1.3, certificate management, PHI protection)
- Long-context reasoning (12 major components with complex interdependencies)

### Opus 4.7 Strengths
- **Deeper domain reasoning** for medical standards & clinical workflows
- **Better architectural planning** for complex multi-component systems
- **Stronger security implementation** for healthcare compliance
- **More thorough error handling** for production hospital environments
- **Superior integration testing** across vendor-specific PACS implementations

## Technical Stack
- **Language**: Python 3.9+
- **DICOM Library**: pynetdicom (already in requirements.txt)
- **Existing Components**: 
  - `src/clinical/dicom_adapter.py` - Basic DICOM support
  - `src/clinical/workflow.py` - Clinical workflow system
  - `src/data/wsi_pipeline/` - WSI processing pipeline
  - `src/clinical/audit.py` - Basic audit logging

## Key Constraints
1. **Security**: All PACS communications use TLS 1.3 encryption
2. **Compliance**: HIPAA audit trails, tamper-evident logging
3. **Performance**: <5s inference, 50 concurrent studies, 10 MB/s transfer
4. **Reliability**: Exponential backoff, dead letter queues, failover
5. **Multi-vendor**: GE, Philips, Siemens, Agfa PACS support

## Property-Based Testing
- 47 property tests defined in tasks.md (marked with `*` as optional)
- Use Hypothesis for correctness validation
- Focus on: DICOM round-trip integrity, error handling, workflow sequencing

## Success Criteria
1. Query PACS for WSI studies (C-FIND)
2. Retrieve WSI files (C-MOVE)
3. Store AI results as DICOM SR (C-STORE)
4. Multi-vendor support (GE/Philips/Siemens/Agfa)
5. TLS 1.3 encryption + certificate validation
6. Automated workflow (poll→retrieve→analyze→store)
7. Clinical notifications (email/SMS/HL7)
8. HIPAA-compliant audit logging
9. <10 min Windows CI (integration tests marked slow)

## Files to Read First
1. `look here ai agents for specs/specs/pacs-integration-system/requirements.md` - Full requirements
2. `look here ai agents for specs/specs/pacs-integration-system/design.md` - Technical design
3. `look here ai agents for specs/specs/pacs-integration-system/tasks.md` - Implementation plan
4. `src/clinical/dicom_adapter.py` - Existing DICOM code
5. `src/clinical/workflow.py` - Existing workflow system

## Recommended Approach
1. **Start with Task 5** (multi-vendor support) - Foundation for everything else
2. **Then Task 6** (error handling) - Critical for production reliability
3. **Then Task 8** (workflow orchestration) - Core value proposition
4. **Then Task 11** (audit logging) - Regulatory compliance
5. **Then Task 10** (notifications) - Clinical integration
6. **Then Task 9** (DICOM parsing) - Enhanced capabilities
7. **Finally Task 13** (integration) - Wire everything together

## Testing Strategy
- Write property tests FIRST (observation-first methodology)
- Run on unfixed code to establish baseline
- Implement fix
- Verify properties hold
- Integration tests for end-to-end workflows

## Business Impact
**This is the gateway to clinical deployment.** Without PACS integration, HistoCore can't be used in real hospitals. It's production-critical for:
- Automated WSI retrieval from hospital imaging systems
- AI analysis result delivery to radiologists/pathologists
- Real-time clinical workflow integration
- Regulatory compliance for FDA/CE approval

## Questions to Ask User
1. Which vendor PACS to prioritize? (GE/Philips/Siemens/Agfa)
2. Hospital network constraints? (firewall rules, VPN, etc.)
3. Notification preferences? (email/SMS/HL7 priority)
4. Audit retention period? (default 7 years for HIPAA)
5. Performance requirements? (concurrent studies, throughput)

## Current State
- **Completed**: Basic PACS adapter components (30%)
- **Remaining**: Production features (70%)
- **Blocker**: None - foundation ready for build-out
- **Timeline**: Estimate 2-3 weeks for Opus 4.7 (complex domain)

## Final Notes
- This is **production-critical** work for clinical deployment
- Requires **deep medical domain knowledge** (DICOM, HL7, PACS)
- Needs **strong security implementation** (TLS, certificates, PHI)
- Demands **thorough error handling** (hospital networks unreliable)
- Must support **multi-vendor PACS** (GE, Philips, Siemens, Agfa)

**You are building the bridge between HistoCore AI and real hospital infrastructure.**

Good luck! 🏥🔬
