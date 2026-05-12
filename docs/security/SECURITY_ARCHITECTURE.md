# Security Architecture

## Overview

This document describes the security architecture of HistoCore, including defense-in-depth layers, security controls, and threat mitigation strategies.

## Architecture Principles

### 1. Defense in Depth

Multiple layers of security controls:

```
┌─────────────────────────────────────────┐
│  Layer 7: Monitoring & Incident Response│
├─────────────────────────────────────────┤
│  Layer 6: Audit & Compliance            │
├─────────────────────────────────────────┤
│  Layer 5: Application Security          │
├─────────────────────────────────────────┤
│  Layer 4: Authentication & Authorization│
├─────────────────────────────────────────┤
│  Layer 3: Network Security              │
├─────────────────────────────────────────┤
│  Layer 2: Infrastructure Security       │
├─────────────────────────────────────────┤
│  Layer 1: Physical Security             │
└─────────────────────────────────────────┘
```

### 2. Least Privilege

- Users have minimum necessary permissions
- Services run with minimal privileges
- Database access is role-based
- API keys are scoped to specific operations

### 3. Fail Secure

- Errors default to deny access
- Validation failures reject input
- Authentication failures lock accounts
- Network failures close connections

### 4. Zero Trust

- Verify every request
- Never trust, always verify
- Assume breach mentality
- Micro-segmentation

## Security Components

### 1. Security Module (`src/security/`)

```
src/security/
├── __init__.py              # Public API
├── models.py                # Security data models
├── exceptions.py            # Security exceptions
├── environment.py           # Environment detection
├── config_manager.py        # Security configuration
├── audit_trail.py           # Audit logging
├── jinja2_security.py       # Template security
├── network_binding.py       # Network binding control
├── model_download.py        # Model download security
├── temp_file.py             # Temp file security
├── pickle_security_control.py  # Pickle deserialization
├── url_fetcher.py           # URL opening security
├── validation.py            # Input validation
└── rate_limit.py            # Rate limiting
```

### 2. Environment-Based Security

```python
# Security policies by environment
SECURITY_POLICIES = {
    "production": {
        "strict_binding": True,
        "require_pinned_models": True,
        "block_untrusted_pickle": True,
        "enforce_https": True,
        "audit_all_operations": True,
    },
    "development": {
        "strict_binding": False,
        "require_pinned_models": False,
        "block_untrusted_pickle": False,
        "enforce_https": False,
        "audit_all_operations": False,
    },
    "research": {
        "strict_binding": False,
        "require_pinned_models": True,
        "block_untrusted_pickle": True,
        "enforce_https": False,
        "audit_all_operations": True,
    },
}
```

### 3. Security Audit Trail

```python
# Audit trail architecture
SecurityAuditTrail
├── JSON structured logging
├── Tamper-evident logging (HMAC)
├── Log rotation and retention
├── Searchable event history
└── Compliance reporting
```

## Threat Model

### STRIDE Analysis

#### Spoofing
- **Threat:** Attacker impersonates legitimate user
- **Mitigation:**
  - Strong authentication (bcrypt password hashing)
  - JWT tokens with expiration
  - API key validation
  - Rate limiting on authentication endpoints

#### Tampering
- **Threat:** Attacker modifies data or code
- **Mitigation:**
  - Input validation (InputValidator)
  - Parameterized SQL queries
  - HMAC signatures on audit logs
  - File integrity monitoring

#### Repudiation
- **Threat:** User denies performing action
- **Mitigation:**
  - Comprehensive audit logging
  - Immutable audit trail
  - Timestamp all operations
  - Log user identity with actions

#### Information Disclosure
- **Threat:** Sensitive data exposed
- **Mitigation:**
  - Encryption at rest (database)
  - Encryption in transit (TLS)
  - Secure error messages
  - Log sanitization
  - Access controls

#### Denial of Service
- **Threat:** Service unavailable
- **Mitigation:**
  - Rate limiting (RateLimiter)
  - Resource quotas
  - Input size limits
  - Timeout controls
  - Load balancing

#### Elevation of Privilege
- **Threat:** Attacker gains unauthorized access
- **Mitigation:**
  - Role-based access control (RBAC)
  - Principle of least privilege
  - Authorization checks on all endpoints
  - Secure session management

## Security Controls

### 1. Input Validation

```python
# Validation architecture
InputValidator
├── Path validation (prevent traversal)
├── File extension validation
├── String validation (pattern matching)
├── Integer/float validation (range checking)
├── Filename sanitization
└── Batch size/threshold validation
```

### 2. Authentication

```python
# Authentication flow
User Request
    ↓
JWT Token Validation
    ↓
Token Expiration Check
    ↓
User Lookup
    ↓
Permission Check
    ↓
Allow/Deny
```

### 3. Authorization

```python
# RBAC model
Roles:
├── admin (full access)
├── pathologist (read/write patient data)
├── researcher (read-only, no PHI)
└── viewer (read-only, limited access)

Permissions:
├── read:patients
├── write:patients
├── read:models
├── write:models
├── admin:users
└── admin:system
```

### 4. Network Security

```python
# Network binding control
NetworkBindingManager
├── Production: 127.0.0.1 only (unless configured)
├── Development: 127.0.0.1 default
├── Research: 0.0.0.0 allowed with warning
└── Audit all binding decisions
```

### 5. Cryptography

```python
# Cryptographic controls
Encryption:
├── Passwords: bcrypt (cost factor 12)
├── Tokens: JWT with HS256
├── Database: AES-256-GCM
├── Transit: TLS 1.2+
└── Audit logs: HMAC-SHA256

Key Management:
├── Environment variables (development)
├── AWS Secrets Manager (production)
├── Azure Key Vault (production)
└── Rotation: 90 days
```

## Data Flow Security

### 1. API Request Flow

```
Client Request
    ↓
[TLS Termination]
    ↓
[Rate Limiting]
    ↓
[Authentication]
    ↓
[Authorization]
    ↓
[Input Validation]
    ↓
[Business Logic]
    ↓
[Output Sanitization]
    ↓
[Audit Logging]
    ↓
Response
```

### 2. File Upload Flow

```
File Upload
    ↓
[Size Validation]
    ↓
[Extension Validation]
    ↓
[MIME Type Validation]
    ↓
[Virus Scanning]
    ↓
[Filename Sanitization]
    ↓
[Path Validation]
    ↓
[Secure Storage]
    ↓
[Audit Logging]
```

### 3. Model Download Flow

```
Model Request
    ↓
[Revision Validation]
    ↓
[Source Validation]
    ↓
[Download with Pinned Revision]
    ↓
[Checksum Verification]
    ↓
[Secure Storage]
    ↓
[Audit Logging]
```

## Deployment Architecture

### Production Deployment

```
┌─────────────────────────────────────────┐
│  Internet                                │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  WAF (Web Application Firewall)         │
│  - OWASP Core Rule Set                  │
│  - Rate limiting                        │
│  - DDoS protection                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Load Balancer (TLS termination)        │
│  - TLS 1.2+                             │
│  - Certificate management               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Application Servers (DMZ)              │
│  - FastAPI application                  │
│  - Security middleware                  │
│  - Audit logging                        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Database (Private subnet)              │
│  - Encrypted at rest                    │
│  - No public access                     │
│  - Backup encryption                    │
└─────────────────────────────────────────┘
```

### Network Segmentation

```
┌─────────────────────────────────────────┐
│  Public Subnet (DMZ)                    │
│  - Load balancer                        │
│  - WAF                                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Application Subnet                     │
│  - API servers                          │
│  - Web servers                          │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Data Subnet (Private)                  │
│  - Database                             │
│  - File storage                         │
│  - No internet access                   │
└─────────────────────────────────────────┘
```

## Compliance

### HIPAA

- **§164.308(a)(1)(ii)(D)**: Audit controls ✓
- **§164.308(a)(3)(i)**: Access management ✓
- **§164.308(a)(4)(i)**: Information access management ✓
- **§164.312(a)(1)**: Access control ✓
- **§164.312(b)**: Audit controls ✓
- **§164.312(c)(1)**: Integrity controls ✓
- **§164.312(d)**: Person or entity authentication ✓
- **§164.312(e)(1)**: Transmission security ✓

### GDPR

- **Article 25**: Privacy by design ✓
- **Article 32**: Security of processing ✓
- **Article 33**: Breach notification ✓
- **Article 35**: Data protection impact assessment ✓

### SOC 2

- **CC6.1**: Logical and physical access controls ✓
- **CC6.6**: Encryption ✓
- **CC6.7**: Transmission security ✓
- **CC7.2**: System monitoring ✓

## Security Metrics

### Key Performance Indicators

```python
SECURITY_METRICS = {
    "authentication": {
        "failed_login_rate": "< 5%",
        "account_lockout_rate": "< 1%",
        "password_reset_rate": "< 10%",
    },
    "authorization": {
        "unauthorized_access_attempts": "0",
        "privilege_escalation_attempts": "0",
    },
    "input_validation": {
        "validation_failure_rate": "< 1%",
        "injection_attempts_blocked": "100%",
    },
    "availability": {
        "uptime": "> 99.9%",
        "rate_limit_violations": "< 0.1%",
    },
    "audit": {
        "audit_log_completeness": "100%",
        "audit_log_integrity": "100%",
    },
}
```

## References

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Application Security Verification Standard](https://owasp.org/www-project-application-security-verification-standard/)
- [CIS Controls](https://www.cisecurity.org/controls)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [GDPR](https://gdpr-info.eu/)
