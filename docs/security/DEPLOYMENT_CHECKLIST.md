# Security Deployment Checklist

## Pre-Deployment Security Checklist

### Environment Configuration
- [ ] JWT_SECRET_KEY set (min 32 bytes)
- [ ] TRUSTED_PROXIES configured with valid IPs
- [ ] ENVIRONMENT set to "production"
- [ ] Database credentials in environment (not hardcoded)
- [ ] API keys in environment (not hardcoded)
- [ ] ALLOWED_ORIGINS configured for CORS
- [ ] Session timeout configured (default: 30 min)

### HTTPS/TLS
- [ ] HTTPS enabled (required for production)
- [ ] Valid TLS certificate installed
- [ ] TLS 1.2+ enforced (disable TLS 1.0/1.1)
- [ ] Strong cipher suites configured
- [ ] HSTS header enabled (Strict-Transport-Security)
- [ ] Certificate auto-renewal configured

### Authentication & Authorization
- [ ] Password complexity requirements enforced
- [ ] Account lockout after failed attempts (5 attempts)
- [ ] Session timeout configured
- [ ] JWT token expiration set (30 minutes)
- [ ] RBAC permissions configured
- [ ] Admin accounts use strong passwords
- [ ] Default credentials changed

### Input Validation
- [ ] All user inputs validated
- [ ] File uploads validated (magic bytes)
- [ ] File size limits enforced (100MB)
- [ ] Path traversal protection enabled
- [ ] SQL injection protection (parameterized queries)
- [ ] XSS protection (output encoding)
- [ ] CSRF protection enabled (if using cookies)

### Rate Limiting
- [ ] Global rate limit configured (30/min)
- [ ] Login rate limit configured (5/min)
- [ ] API endpoint rate limits set
- [ ] Redis configured for distributed rate limiting
- [ ] Rate limit monitoring enabled

### Security Headers
- [ ] Content-Security-Policy configured
- [ ] X-Frame-Options: DENY
- [ ] X-Content-Type-Options: nosniff
- [ ] X-XSS-Protection: 1; mode=block
- [ ] Referrer-Policy configured
- [ ] Permissions-Policy configured

### Data Protection
- [ ] PHI encryption at rest (AES-256)
- [ ] PHI encryption in transit (TLS)
- [ ] Database encryption enabled
- [ ] Backup encryption enabled
- [ ] Secure key management (not in code)
- [ ] Key rotation schedule defined

### Logging & Monitoring
- [ ] Security audit logging enabled
- [ ] Failed login attempts logged
- [ ] PHI access logged
- [ ] Attack indicators logged
- [ ] Log retention configured (7 years for HIPAA)
- [ ] Log integrity protection enabled
- [ ] Monitoring alerts configured

### Dependency Security
- [ ] All dependencies up to date
- [ ] Vulnerability scan passed (safety, pip-audit)
- [ ] No known CVEs in dependencies
- [ ] Dependency pinning enabled
- [ ] Regular dependency updates scheduled

### Code Security
- [ ] No hardcoded secrets
- [ ] No SQL injection vulnerabilities
- [ ] No command injection vulnerabilities
- [ ] No path traversal vulnerabilities
- [ ] No XXE vulnerabilities (use defusedxml)
- [ ] No insecure deserialization (pickle)
- [ ] No ReDoS vulnerabilities

### Infrastructure
- [ ] Firewall configured
- [ ] Database not publicly accessible
- [ ] Redis not publicly accessible
- [ ] SSH key-based authentication only
- [ ] Unnecessary services disabled
- [ ] OS security updates applied
- [ ] Container security scanning passed

### Compliance
- [ ] HIPAA compliance verified
- [ ] Audit trail enabled
- [ ] Patient consent management configured
- [ ] Data retention policies configured
- [ ] Breach notification procedures documented
- [ ] Privacy policy published
- [ ] Terms of service published

### Backup & Recovery
- [ ] Automated backups configured
- [ ] Backup encryption enabled
- [ ] Backup restoration tested
- [ ] Disaster recovery plan documented
- [ ] RTO/RPO defined and tested
- [ ] Backup retention policy configured

### Testing
- [ ] Security tests passed
- [ ] Penetration testing completed
- [ ] Vulnerability assessment completed
- [ ] Load testing completed
- [ ] Failover testing completed

## Post-Deployment Verification

### Immediate (Day 1)
- [ ] HTTPS working correctly
- [ ] Authentication working
- [ ] Rate limiting working
- [ ] Security headers present
- [ ] Logs being written
- [ ] Monitoring alerts working

### Week 1
- [ ] Review security logs
- [ ] Check for failed login attempts
- [ ] Verify rate limit violations
- [ ] Review error logs
- [ ] Check system performance

### Month 1
- [ ] Security audit review
- [ ] Dependency vulnerability scan
- [ ] Access control review
- [ ] Backup restoration test
- [ ] Incident response drill

## Ongoing Maintenance

### Weekly
- [ ] Review security logs
- [ ] Check for suspicious activity
- [ ] Monitor rate limit violations

### Monthly
- [ ] Dependency vulnerability scan
- [ ] Review access logs
- [ ] Update dependencies
- [ ] Review user accounts

### Quarterly
- [ ] Security audit
- [ ] Penetration testing
- [ ] Disaster recovery test
- [ ] Compliance review

### Annually
- [ ] Full security assessment
- [ ] Third-party security audit
- [ ] Compliance certification renewal
- [ ] Update security policies

## Incident Response

### Detection
- [ ] Monitoring alerts configured
- [ ] Log analysis automated
- [ ] Anomaly detection enabled

### Response
- [ ] Incident response plan documented
- [ ] Contact list maintained
- [ ] Escalation procedures defined
- [ ] Communication templates prepared

### Recovery
- [ ] Backup restoration procedures tested
- [ ] Failover procedures documented
- [ ] Post-incident review process defined

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/)
- [CIS Controls](https://www.cisecurity.org/controls/)
