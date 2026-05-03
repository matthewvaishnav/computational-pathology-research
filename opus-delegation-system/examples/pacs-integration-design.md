# Example: PACS Integration Design

## Problem Description

Design integration layer for PACS (Picture Archiving and Communication System) using DICOM protocol. Need to query studies, retrieve images, handle large WSI (Whole Slide Images), implement retry logic, and ensure HIPAA compliance.

## Classification

```typescript
const classifier = new ProblemClassifier();
const result = classifier.classifyProblem(
  'Design PACS integration with DICOM protocol for WSI retrieval'
);

// Result:
// {
//   shouldDelegate: true,
//   classification: {
//     delegationType: 'integration_design',
//     complexity: 'complex',
//     requiredContext: [
//       'existing_dicom_code',
//       'pacs_specifications',
//       'error_logs',
//       'retry_policies'
//     ]
//   },
//   recommendation: 'Complex protocol integration with reliability requirements. Opus delegation recommended.'
// }
```

## Context Bundle

```markdown
# Context Bundle: PACS Integration Design

## Problem Summary
Design integration layer for PACS using DICOM protocol. Must handle study queries, image retrieval, large WSI files (>1GB), implement robust retry logic, ensure HIPAA compliance.

## Relevant Code

### src/dicom/client.py (lines 10-60)
```python
import pydicom
from pynetdicom import AE, evt

class DICOMClient:
    """Basic DICOM client for PACS communication"""
    def __init__(self, host, port, ae_title):
        self.host = host
        self.port = port
        self.ae_title = ae_title
        self.ae = AE(ae_title=ae_title)
    
    def query_studies(self, patient_id):
        """Query studies for patient"""
        # C-FIND request
        assoc = self.ae.associate(self.host, self.port)
        if assoc.is_established:
            responses = assoc.send_c_find(
                dataset=self._build_query(patient_id),
                query_model='StudyRoot'
            )
            studies = [r for r in responses if r[0].Status == 0x0000]
            assoc.release()
            return studies
        else:
            raise ConnectionError('Failed to establish association')
    
    def retrieve_image(self, study_uid, series_uid, instance_uid):
        """Retrieve single DICOM instance"""
        # C-MOVE request
        assoc = self.ae.associate(self.host, self.port)
        if assoc.is_established:
            responses = assoc.send_c_move(
                dataset=self._build_retrieve_query(
                    study_uid, series_uid, instance_uid
                ),
                move_aet=self.ae_title
            )
            assoc.release()
            return responses
        else:
            raise ConnectionError('Failed to establish association')
```

### src/storage/wsi_handler.py (lines 15-45)
```python
class WSIHandler:
    """Handler for whole slide images"""
    def __init__(self, storage_path):
        self.storage_path = storage_path
    
    def store_wsi(self, dicom_file, study_uid):
        """Store WSI file"""
        # WSI files can be >1GB, need streaming
        output_path = os.path.join(
            self.storage_path,
            study_uid,
            dicom_file.SOPInstanceUID + '.dcm'
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Currently loads entire file into memory - problematic for WSI
        dicom_file.save_as(output_path)
        return output_path
```

## Documentation Excerpts

**From:** docs/pacs_spec.md
PACS supports DICOM 3.0. Query/Retrieve service class. C-FIND for queries, C-MOVE for retrieval. AE Title: PACS_SERVER. Port: 11112. Supports StudyRoot and PatientRoot query models.

**From:** docs/error_logs.md
Common errors:
- Connection timeout after 30s (network issues)
- Association rejected (wrong AE title)
- C-MOVE failures for large files (>500MB)
- Incomplete transfers (connection drops mid-transfer)

**From:** docs/compliance.md
HIPAA requirements: Audit all DICOM queries and retrievals. Log patient ID, timestamp, user. Encrypt data in transit (TLS). Encrypt data at rest. Retain audit logs for 7 years.

## Constraints
- PACS connection: 100 Mbps, 30s timeout
- WSI files: 500MB - 2GB per file
- Concurrent requests: max 5 per PACS
- Retry policy: 3 attempts with exponential backoff
- Must support both C-MOVE and C-GET
- TLS 1.2+ required for production

## Context Manifest

| Source | Type | Size | Relevance |
|--------|------|------|-----------|
| src/dicom/client.py | Code | 2.1 KB | High |
| src/storage/wsi_handler.py | Code | 1.5 KB | High |
| docs/pacs_spec.md | Doc | 1.0 KB | High |
| docs/error_logs.md | Doc | 0.8 KB | High |
| docs/compliance.md | Doc | 0.6 KB | Medium |
```

## Delegation Request

```markdown
# Delegation Request: PACS Integration Design

## Objective
Design a robust integration layer for PACS (Picture Archiving and Communication System) using DICOM protocol. The system must handle study queries, image retrieval (including large WSI files >1GB), implement comprehensive retry logic, ensure HIPAA compliance, and provide high reliability for clinical workflows.

## Background
Current implementation has basic DICOM query/retrieve but lacks:
- Robust error handling and retry logic
- Streaming support for large WSI files (currently loads entire file into memory)
- Connection pooling and rate limiting
- Comprehensive audit logging
- TLS encryption for production

Clinical workflows depend on reliable PACS access. Failures cause delays in diagnosis.

## Expected Artifacts
Please generate the following:

1. **Integration Architecture Diagram** — Mermaid diagram showing:
   - DICOM client components (query, retrieve, store)
   - Connection management (pooling, rate limiting)
   - Retry and error handling mechanisms
   - Audit logging pipeline
   - Storage layer for retrieved images
   - Data flow for query and retrieve operations

2. **API Specification** — OpenAPI 3.0 YAML for:
   - Study query endpoint
   - Image retrieval endpoint (single instance and bulk)
   - WSI streaming endpoint
   - Audit log query endpoint
   - Health check and status endpoints
   - Error responses and retry guidance

3. **Implementation Plan** — Detailed steps including:
   - Phase 1: Enhanced DICOM client with connection pooling
   - Phase 2: Streaming WSI handler
   - Phase 3: Retry logic and circuit breaker
   - Phase 4: Audit logging system
   - Phase 5: TLS integration
   - Dependencies and complexity estimates

## Output Format Requirements
- Diagrams: Use Mermaid syntax in fenced code blocks (```mermaid)
- API specs: Use OpenAPI 3.0 YAML in fenced code blocks (```yaml)
- Plans: Use markdown numbered lists with sub-items for dependencies

## Context
[Context Bundle inserted here]

## Questions to Address
1. How to implement streaming for large WSI files (>1GB)?
2. What retry strategy for different DICOM error types?
3. How to implement connection pooling for PACS?
4. How to handle partial transfers and resume?
5. What circuit breaker pattern for PACS failures?
6. How to structure audit logs for HIPAA compliance?
7. How to implement rate limiting (max 5 concurrent requests)?
8. What caching strategy for frequently accessed studies?

## Success Criteria
- Support WSI files up to 2GB without memory issues
- Retry logic handles transient failures (3 attempts, exponential backoff)
- Connection pool maintains 5 concurrent connections
- Audit logs capture all required HIPAA fields
- TLS 1.2+ encryption for production
- 99.9% reliability for DICOM operations
- Graceful degradation when PACS unavailable
```

## Opus Response (Sample)

```markdown
# PACS Integration Design

## Integration Architecture

```mermaid
graph TB
    subgraph "API Layer"
        API[REST API]
        WS[WebSocket for Progress]
    end
    
    subgraph "DICOM Client Layer"
        QM[Query Manager]
        RM[Retrieve Manager]
        CP[Connection Pool]
        RL[Rate Limiter]
        
        QM --> CP
        RM --> CP
        CP --> RL
    end
    
    subgraph "Reliability Layer"
        RH[Retry Handler]
        CB[Circuit Breaker]
        QU[Request Queue]
        
        RL --> RH
        RH --> CB
        CB --> QU
    end
    
    subgraph "PACS"
        PACS[PACS Server]
        QU --> PACS
    end
    
    subgraph "Storage Layer"
        SM[Stream Manager]
        CH[Chunk Handler]
        FS[File System]
        
        PACS --> SM
        SM --> CH
        CH --> FS
    end
    
    subgraph "Audit & Monitoring"
        AL[Audit Logger]
        MT[Metrics Tracker]
        
        QM --> AL
        RM --> AL
        RH --> MT
        CB --> MT
    end
    
    API --> QM
    API --> RM
    RM --> SM
    SM --> WS
```

## API Specification

```yaml
openapi: 3.0.0
info:
  title: PACS Integration API
  version: 1.0.0
  description: API for PACS integration with DICOM protocol

servers:
  - url: https://pacs-integration.example.com/api/v1

paths:
  /studies/query:
    post:
      summary: Query studies from PACS
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                patient_id:
                  type: string
                study_date_range:
                  type: object
                  properties:
                    start:
                      type: string
                      format: date
                    end:
                      type: string
                      format: date
                modality:
                  type: string
                  enum: [CT, MR, CR, DX, MG, US, PT, SM]
              required:
                - patient_id
      responses:
        '200':
          description: Studies found
          content:
            application/json:
              schema:
                type: object
                properties:
                  studies:
                    type: array
                    items:
                      $ref: '#/components/schemas/Study'
                  total_count:
                    type: integer
        '400':
          description: Invalid query parameters
        '503':
          description: PACS unavailable
          headers:
            Retry-After:
              schema:
                type: integer
              description: Seconds to wait before retry
  
  /images/retrieve:
    post:
      summary: Retrieve DICOM image
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                study_uid:
                  type: string
                series_uid:
                  type: string
                instance_uid:
                  type: string
                stream:
                  type: boolean
                  default: false
                  description: Use streaming for large files
              required:
                - study_uid
                - series_uid
                - instance_uid
      responses:
        '202':
          description: Retrieval initiated
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                  status_url:
                    type: string
                  websocket_url:
                    type: string
                    description: WebSocket for progress updates
        '400':
          description: Invalid UIDs
        '503':
          description: PACS unavailable
  
  /images/retrieve/{job_id}/status:
    get:
      summary: Get retrieval job status
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Job status
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                  status:
                    type: string
                    enum: [pending, in_progress, completed, failed]
                  progress:
                    type: number
                    format: float
                    description: Progress percentage (0-100)
                  file_url:
                    type: string
                    description: Download URL (when completed)
                  error:
                    type: string
                    description: Error message (when failed)
  
  /images/stream/{job_id}:
    get:
      summary: Stream large DICOM file
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
        - name: Range
          in: header
          schema:
            type: string
          description: Byte range for partial content
      responses:
        '200':
          description: Full file
          content:
            application/dicom:
              schema:
                type: string
                format: binary
        '206':
          description: Partial content
          headers:
            Content-Range:
              schema:
                type: string
          content:
            application/dicom:
              schema:
                type: string
                format: binary
  
  /audit/logs:
    get:
      summary: Query audit logs
      parameters:
        - name: start_date
          in: query
          schema:
            type: string
            format: date-time
        - name: end_date
          in: query
          schema:
            type: string
            format: date-time
        - name: patient_id
          in: query
          schema:
            type: string
        - name: user_id
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Audit logs
          content:
            application/json:
              schema:
                type: object
                properties:
                  logs:
                    type: array
                    items:
                      $ref: '#/components/schemas/AuditLog'
  
  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    enum: [healthy, degraded, unhealthy]
                  pacs_connection:
                    type: boolean
                  circuit_breaker_state:
                    type: string
                    enum: [closed, open, half_open]

components:
  schemas:
    Study:
      type: object
      properties:
        study_uid:
          type: string
        patient_id:
          type: string
        study_date:
          type: string
          format: date
        modality:
          type: string
        description:
          type: string
        series_count:
          type: integer
    
    AuditLog:
      type: object
      properties:
        timestamp:
          type: string
          format: date-time
        operation:
          type: string
          enum: [query, retrieve]
        patient_id:
          type: string
        user_id:
          type: string
        study_uid:
          type: string
        status:
          type: string
          enum: [success, failure]
        error_message:
          type: string

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
```

## Implementation Plan

### Phase 1: Enhanced DICOM Client (Complexity: Medium)

1. **Create connection pool manager**
   - Implement pool with max 5 connections
   - Add connection health checks
   - Add connection reuse logic
   - Dependencies: None
   - Estimate: 2 days

2. **Implement rate limiter**
   - Token bucket algorithm (5 tokens, 1 refill/sec)
   - Queue requests when limit exceeded
   - Dependencies: Step 1
   - Estimate: 1 day

3. **Enhance query manager**
   - Support StudyRoot and PatientRoot
   - Add query result pagination
   - Add query timeout handling
   - Dependencies: Step 1
   - Estimate: 2 days

### Phase 2: Streaming WSI Handler (Complexity: High)

4. **Implement streaming retrieve**
   - Use C-GET with chunked transfer
   - Write chunks directly to disk (no memory buffering)
   - Support resume from partial transfer
   - Dependencies: Step 1
   - Estimate: 4 days

5. **Create chunk handler**
   - Validate DICOM chunks
   - Assemble chunks into complete file
   - Verify file integrity (checksum)
   - Dependencies: Step 4
   - Estimate: 2 days

6. **Implement progress tracking**
   - Track bytes received vs total
   - Publish progress via WebSocket
   - Dependencies: Step 4
   - Estimate: 1 day

### Phase 3: Retry Logic and Circuit Breaker (Complexity: Medium)

7. **Implement retry handler**
   - Exponential backoff (1s, 2s, 4s)
   - Retry on transient errors (timeout, connection reset)
   - No retry on permanent errors (invalid UID, auth failure)
   - Dependencies: None
   - Estimate: 2 days

8. **Implement circuit breaker**
   - Open after 5 consecutive failures
   - Half-open after 30s
   - Close after 3 successful requests
   - Dependencies: Step 7
   - Estimate: 2 days

9. **Create request queue**
   - Queue requests when circuit open
   - Prioritize by urgency (clinical vs research)
   - Dependencies: Step 8
   - Estimate: 1 day

### Phase 4: Audit Logging (Complexity: Low)

10. **Create audit logger**
    - Log all DICOM operations (query, retrieve)
    - Include HIPAA required fields (patient ID, user, timestamp)
    - Write to structured log format (JSON)
    - Dependencies: None
    - Estimate: 2 days

11. **Implement log storage**
    - Store logs in database (7 year retention)
    - Add log query API
    - Add log export (CSV for compliance audits)
    - Dependencies: Step 10
    - Estimate: 2 days

### Phase 5: TLS Integration (Complexity: Low)

12. **Add TLS support**
    - Configure TLS 1.2+ for DICOM connections
    - Certificate validation
    - Mutual TLS for production
    - Dependencies: Step 1
    - Estimate: 2 days

### Phase 6: Testing (Complexity: Medium)

13. **Write integration tests**
    - Test query/retrieve with real PACS
    - Test retry logic with simulated failures
    - Test streaming with large files (>1GB)
    - Test circuit breaker state transitions
    - Dependencies: All previous steps
    - Estimate: 3 days

## Key Design Decisions

1. **Streaming Strategy:** C-GET with chunked transfer
   - Write chunks directly to disk (no memory buffering)
   - Support HTTP Range requests for client-side streaming
   - Resume from partial transfer using byte offset

2. **Retry Logic:** Exponential backoff with error classification
   - Transient errors: retry (timeout, connection reset)
   - Permanent errors: no retry (invalid UID, auth failure)
   - Max 3 attempts, backoff: 1s, 2s, 4s

3. **Circuit Breaker:** Fail-fast pattern
   - Open after 5 consecutive failures
   - Half-open after 30s (test with single request)
   - Close after 3 successful requests
   - Prevents cascading failures

4. **Connection Pooling:** Fixed pool size
   - Max 5 connections (PACS limit)
   - Health checks every 60s
   - Reuse connections for multiple requests
   - Close idle connections after 5 minutes

5. **Audit Logging:** Structured JSON logs
   - All DICOM operations logged
   - HIPAA required fields: patient ID, user, timestamp, operation, status
   - 7 year retention in database
   - Export to CSV for compliance audits

## Risk Register

| Risk | Mitigation | Owner |
|------|------------|-------|
| PACS connection failures | Circuit breaker, retry logic, request queue | Infrastructure |
| WSI files exceed memory | Streaming with chunked transfer, disk buffering | Backend Team |
| Partial transfers | Resume support, chunk validation, integrity checks | Backend Team |
| Rate limit exceeded | Token bucket rate limiter, request queue | Backend Team |
| Audit log loss | Redundant storage, log shipping to SIEM | Security Team |
| TLS certificate expiry | Automated renewal, monitoring alerts | DevOps |
```

## Validation Results

```typescript
const validator = new ArtifactValidator();
const results = validator.validateAll(artifacts);

// Result:
// [
//   {
//     artifactId: 'artifact-1',
//     artifactType: 'mermaid_diagram',
//     isValid: true,
//     completenessScore: 93,
//     issues: [],
//     suggestions: ['Add error handling flows for PACS failures']
//   },
//   {
//     artifactId: 'artifact-2',
//     artifactType: 'openapi_specification',
//     isValid: true,
//     completenessScore: 95,
//     issues: [],
//     suggestions: ['Add pagination for audit logs endpoint']
//   },
//   {
//     artifactId: 'artifact-3',
//     artifactType: 'implementation_plan',
//     isValid: true,
//     completenessScore: 94,
//     issues: [],
//     suggestions: ['Add performance testing phase']
//   }
// ]
```

## Lessons Learned

1. **Domain expertise matters** — Including DICOM error logs and PACS specs led to practical retry strategies
2. **Streaming is complex** — Opus provided detailed chunked transfer approach with resume support
3. **Compliance requirements** — HIPAA audit logging requirements fully addressed in design
4. **Circuit breaker pattern** — Opus recommended fail-fast pattern with specific thresholds
5. **Real-world constraints** — 5 concurrent connection limit and 30s timeout informed connection pooling design
