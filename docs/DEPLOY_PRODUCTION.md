# Production Deployment Security Checklist

## Pre-Deployment

### Environment Configuration

- [ ] Set `ENVIRONMENT=production` environment variable
- [ ] Set `DEPLOYMENT_ENV=production` environment variable
- [ ] Verify environment detection: `python -c "from src.security.environment import SecurityEnvironment; print(SecurityEnvironment.detect())"`

### Credentials

- [ ] All credentials loaded from environment variables (no hardcoded values)
- [ ] `DB_PASSWORD` set
- [ ] `DB_USER` set
- [ ] `DB_HOST` set
- [ ] `DB_NAME` set
- [ ] `SECRET_KEY` set (for JWT tokens)
- [ ] `ENCRYPTION_PASSWORD` set (for data encryption)
- [ ] SMTP credentials set (if using email notifications)
- [ ] SMS gateway credentials set (if using SMS notifications)

### Security Configuration

- [ ] `config/security_config.yaml` reviewed and customized for production
- [ ] `config/model_revisions.yaml` contains pinned revisions for all models
- [ ] Network binding policy: `strict_binding: true`
- [ ] Model download policy: `require_pinned_revisions: true`
- [ ] Pickle policy: `block_untrusted: true`

### Security Verification

- [ ] Run Bandit scan: `python -m bandit -r src/ -f json -o bandit-report.json`
- [ ] Check results: `python scripts/check_bandit_results.py bandit-report.json`
- [ ] Verify 0 HIGH severity issues
- [ ] Verify 0 MEDIUM severity issues
- [ ] Run security verification: `python scripts/verify_security.py`

### Network Security

- [ ] Firewall configured to block public access to internal services
- [ ] HTTPS/TLS enabled for all external endpoints
- [ ] SSL certificates valid and not expired
- [ ] Database connections use SSL/TLS
- [ ] API rate limiting enabled

### Dependency Security

- [ ] Run `safety check` to scan for known vulnerabilities
- [ ] Run `pip-audit` to check dependency vulnerabilities
- [ ] All dependencies pinned to specific versions in `requirements.txt`
- [ ] No GPL-licensed dependencies (if commercial deployment)

## Deployment

### Application Configuration

- [ ] Debug mode disabled: `DEBUG=false`
- [ ] Logging level set to `INFO` or `WARNING`
- [ ] Audit logging enabled
- [ ] Security audit trail configured with tamper-evident logging

### Server Configuration

- [ ] Web server (Gunicorn/uWSGI) configured with appropriate worker count
- [ ] Reverse proxy (Nginx/Apache) configured with security headers
- [ ] CORS policy configured (if needed)
- [ ] CSRF protection enabled
- [ ] Rate limiting configured

### Database Configuration

- [ ] Database user has minimum required privileges
- [ ] Database backups configured
- [ ] Database encryption at rest enabled
- [ ] Database connection pooling configured

## Post-Deployment

### Verification

- [ ] Run smoke tests: `pytest tests/smoke/ -v`
- [ ] Verify API endpoints respond correctly
- [ ] Verify authentication works
- [ ] Verify database connections work
- [ ] Check logs for errors or warnings

### Monitoring

- [ ] Application monitoring configured (Prometheus/Grafana)
- [ ] Error tracking configured (Sentry)
- [ ] Security audit logs being collected
- [ ] Alerts configured for security events

### Documentation

- [ ] Deployment date and version documented
- [ ] Configuration changes documented
- [ ] Known issues documented
- [ ] Rollback procedure documented

## Security Incident Response

- [ ] Incident response team identified
- [ ] Escalation procedures documented
- [ ] Security contact information updated
- [ ] Incident response playbook reviewed

## Maintenance

### Weekly

- [ ] Review security audit logs
- [ ] Check for failed authentication attempts
- [ ] Review rate limiting logs

### Monthly

- [ ] Run comprehensive security audit: `python scripts/security_posture_report.py`
- [ ] Update dependencies: `pip list --outdated`
- [ ] Review and rotate credentials
- [ ] Review access logs

### Quarterly

- [ ] Security penetration testing
- [ ] Dependency vulnerability assessment
- [ ] Security policy review
- [ ] Incident response drill

## Rollback Procedure

If security issues are discovered post-deployment:

1. **Immediate**: Take affected services offline
2. **Assess**: Determine scope and impact
3. **Rollback**: Revert to previous known-good version
4. **Investigate**: Analyze logs and audit trail
5. **Fix**: Apply security patches
6. **Test**: Verify fixes in staging environment
7. **Redeploy**: Deploy fixed version
8. **Monitor**: Watch for recurrence

## Support

For security issues or questions:
- Email: [security contact]
- Slack: #security-team
- On-call: [on-call rotation]
