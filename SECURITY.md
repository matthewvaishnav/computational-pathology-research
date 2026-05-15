# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by emailing the maintainer. Do not open a public issue.

## Security Measures

This research framework has undergone security hardening with 39+ commits addressing:

### Authentication & Authorization
- JWT token validation
- WebSocket authentication with origin validation
- Input sanitization (username, email, password)
- Admin role verification

### Input Validation
- Pydantic models for API request validation
- File size limits (DICOM uploads)
- Path traversal protection
- Array bounds checking
- Slide ID format validation

### Network Security
- HTTPS enforcement
- SMTP STARTTLS
- Connection pooling with retry limits
- Timeout enforcement
- Rate limiting (100 req/min per IP)

### Database Security
- Parameterized SQL queries (SQLAlchemy text())
- Connection pooling (pool_size=10, max_overflow=20)
- Graceful shutdown with resource cleanup

### Privacy Guarantees (PathologyFL)
- TenSEAL required for homomorphic encryption
- Opacus required for differential privacy accounting
- No silent degradation to plaintext
- Proper dataset size in noise calibration

## Known Limitations

This is a research framework. Before clinical deployment:

1. Conduct independent security audit
2. Implement comprehensive logging and monitoring
3. Set up intrusion detection
4. Configure proper secrets management
5. Enable audit trails for HIPAA compliance
6. Test disaster recovery procedures

## Bandit Security Scan Results

**Last Scan**: 2026-05-15T17:11:50Z  
**Report**: `bandit-final-clean.json`

### Summary
- **HIGH Severity**: 0 issues ✅
- **MEDIUM Severity**: 0 issues ✅
- **LOW Severity**: 195 issues (expected/justified)
- **Lines of Code Scanned**: 146,893

### Security Patterns Explained

All remaining LOW severity findings are false positives or acceptable patterns:

#### subprocess Usage (B603, B607)
- **Context**: Used in analysis/deployment modules for legitimate system operations
- **Mitigation**: Input validation, no user-controlled shell injection
- **Justification**: Required for build/test automation

#### Standard Pseudo-Random (B311)
- **Context**: Used for non-cryptographic purposes (sampling, shuffling)
- **Mitigation**: `secrets` module used for all cryptographic operations
- **Justification**: `random` module appropriate for ML/statistical tasks

#### pickle Usage (B301)
- **Context**: Cache deserialization in `src/utils/caching.py`
- **Mitigation**: HMAC validation + safe_pickle wrapper with restricted unpickler
- **Justification**: Double-layer security prevents malicious pickle attacks

#### hardcoded_bind_all_interfaces (B104)
- **Context**: Development server binding in CLI
- **Mitigation**: Marked with `# nosec B104` - dev only, not production
- **Justification**: Intentional for local development

### Fixed Issues

#### Medium Severity (Fixed)
1. **B113**: Missing timeout in `requests.post()` → Added 30s default timeout
2. **B301**: Direct `pickle.loads()` → Replaced with `safe_pickle.loads(trusted=True)`

#### High Severity (Already Fixed)
1. **B201**: Jinja2 XSS → `autoescape=True` enabled in all templates

## Security Best Practices

When deploying:

- Use environment variables for secrets (never commit credentials)
- Enable HTTPS with valid certificates
- Configure firewall rules
- Use Redis for production rate limiting
- Set up proper backup procedures
- Monitor for security updates in dependencies

## Compliance

This software is for research purposes only. It is not FDA-approved or CE-marked for clinical diagnostic use. Any clinical deployment requires:

- Appropriate regulatory approval
- Clinical validation studies
- HIPAA compliance verification
- Security audit by qualified professionals
