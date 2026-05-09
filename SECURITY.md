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
