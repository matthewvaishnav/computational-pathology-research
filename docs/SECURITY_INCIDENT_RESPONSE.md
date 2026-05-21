# Security Incident Response Guide

## Incident Classification

### Severity Levels

**CRITICAL** (P0):
- Active data breach
- Unauthorized access to patient data
- Ransomware/malware infection
- Complete service outage due to security incident

**HIGH** (P1):
- Suspected data breach
- Privilege escalation vulnerability
- Authentication bypass
- Partial service outage due to security incident

**MEDIUM** (P2):
- Failed authentication attempts (brute force)
- Suspicious activity in logs
- Non-critical vulnerability discovered
- Security policy violation

**LOW** (P3):
- Security configuration drift
- Outdated dependencies with known vulnerabilities
- Security best practice violations

## Incident Response Procedure

### 1. Detection and Identification

**Indicators of Compromise (IoCs)**:
- Multiple failed authentication attempts
- Unusual API access patterns
- Unexpected data exports
- Suspicious file modifications
- Abnormal network traffic
- Security audit log anomalies

**Detection Sources**:
- Security audit logs (`logs/security_audit.log`)
- Application logs
- Monitoring alerts (Prometheus/Grafana)
- Error tracking (Sentry)
- User reports

### 2. Containment

**Immediate Actions** (within 15 minutes):

1. **Assess scope**: Determine affected systems and data
2. **Isolate affected systems**: Take offline if necessary
3. **Preserve evidence**: Copy logs and audit trails
4. **Notify stakeholders**: Alert security team and management

**Containment Commands**:
```bash
# Stop affected service
systemctl stop the platform-api

# Block suspicious IP
iptables -A INPUT -s <suspicious-ip> -j DROP

# Revoke compromised credentials
python scripts/revoke_credentials.py --user <username>

# Enable enhanced logging
export LOG_LEVEL=DEBUG
```

### 3. Investigation

**Analyze Security Audit Logs**:
```bash
# Search for failed authentication attempts
grep "LOGIN_FAILURE" logs/security_audit.log | tail -100

# Search for unauthorized access attempts
grep "AUTHORIZATION_DENIED" logs/security_audit.log | tail -100

# Search for suspicious operations
grep "SECURITY_WARNING" logs/security_audit.log | tail -100
```

**Check for Compromised Credentials**:
```bash
# Review recent authentication events
python scripts/analyze_auth_events.py --since "2024-01-01"

# Check for privilege escalation
python scripts/check_privilege_changes.py --since "2024-01-01"
```

**Analyze Network Traffic**:
```bash
# Review API access logs
grep "POST /api/inference" logs/api.log | awk '{print $1}' | sort | uniq -c | sort -rn

# Check for data exfiltration
grep "GET /api/patients" logs/api.log | grep -v "200" | tail -100
```

### 4. Eradication

**Remove Threat**:
1. **Patch vulnerabilities**: Apply security updates
2. **Remove malware**: Scan and clean infected systems
3. **Close attack vectors**: Fix configuration issues
4. **Rotate credentials**: Change all potentially compromised passwords/keys

**Credential Rotation**:
```bash
# Rotate database password
export NEW_DB_PASSWORD=$(openssl rand -base64 32)
python scripts/rotate_db_password.py --new-password "$NEW_DB_PASSWORD"

# Rotate API keys
python scripts/rotate_api_keys.py --all

# Rotate JWT secret
export NEW_SECRET_KEY=$(openssl rand -base64 64)
python scripts/rotate_jwt_secret.py --new-secret "$NEW_SECRET_KEY"
```

### 5. Recovery

**Restore Services**:
1. **Verify fixes**: Test in staging environment
2. **Deploy patches**: Apply fixes to production
3. **Restore from backup**: If data corruption occurred
4. **Restart services**: Bring systems back online
5. **Monitor closely**: Watch for recurrence

**Recovery Commands**:
```bash
# Verify security posture
python scripts/verify_security.py

# Run security tests
pytest tests/security/ -v

# Restart services
systemctl start the platform-api

# Monitor logs
tail -f logs/security_audit.log
```

### 6. Post-Incident Review

**Within 48 hours**:
1. **Document timeline**: Record all events and actions
2. **Root cause analysis**: Identify how incident occurred
3. **Impact assessment**: Determine data/systems affected
4. **Lessons learned**: Identify improvements
5. **Update procedures**: Revise incident response plan

**Post-Incident Report Template**:
```markdown
# Security Incident Report

## Incident Summary
- **Date/Time**: [timestamp]
- **Severity**: [P0/P1/P2/P3]
- **Status**: [Resolved/Ongoing]
- **Affected Systems**: [list]

## Timeline
- [timestamp]: Incident detected
- [timestamp]: Containment actions taken
- [timestamp]: Investigation completed
- [timestamp]: Threat eradicated
- [timestamp]: Services restored

## Root Cause
[Description of how incident occurred]

## Impact
- **Data**: [affected data]
- **Systems**: [affected systems]
- **Users**: [affected users]
- **Duration**: [downtime duration]

## Actions Taken
1. [action 1]
2. [action 2]
...

## Lessons Learned
- [lesson 1]
- [lesson 2]
...

## Recommendations
1. [recommendation 1]
2. [recommendation 2]
...
```

## Escalation Procedures

### Internal Escalation

**Level 1** (Security Team):
- Email: security-team@company.com
- Slack: #security-incidents
- On-call: [phone number]

**Level 2** (Engineering Leadership):
- CTO: [contact]
- VP Engineering: [contact]

**Level 3** (Executive Leadership):
- CEO: [contact]
- Legal: [contact]

### External Escalation

**Regulatory Reporting** (if patient data affected):
- HIPAA breach notification: Within 60 days
- State breach notification laws: Varies by state
- OCR breach portal: https://ocrportal.hhs.gov/

**Law Enforcement** (if criminal activity):
- FBI Cyber Division: https://www.fbi.gov/investigate/cyber
- Local law enforcement: [contact]

## Communication Templates

### Internal Notification

```
Subject: [P0/P1/P2/P3] Security Incident - [Brief Description]

Team,

A security incident has been detected:

Severity: [P0/P1/P2/P3]
Affected Systems: [list]
Status: [Investigating/Contained/Resolved]
Impact: [description]

Actions Taken:
- [action 1]
- [action 2]

Next Steps:
- [step 1]
- [step 2]

Updates will be provided every [frequency].

[Your Name]
Security Team
```

### Customer Notification (if required)

```
Subject: Security Incident Notification

Dear [Customer],

I am writing to inform you of a security incident that may have affected your data.

What Happened:
[Brief description]

What Information Was Involved:
[List of affected data types]

What Actions Have Been Taken:
[Actions taken to address incident]

What You Can Do:
[Recommended actions for customers]

For More Information:
Contact: security@company.com
Phone: [number]

I sincerely apologize for this incident and am committed to protecting your data.

[Company Name]
```

## Prevention Measures

### Proactive Security

1. **Regular security audits**: Weekly log reviews
2. **Vulnerability scanning**: Monthly Bandit + Safety scans
3. **Penetration testing**: Quarterly external testing
4. **Security training**: Annual training for all staff
5. **Incident response drills**: Quarterly tabletop exercises

### Monitoring and Alerting

**Critical Alerts**:
- Multiple failed authentication attempts (>5 in 5 minutes)
- Unauthorized access attempts
- Privilege escalation events
- Suspicious data exports (>1000 records)
- Service outages
- Security configuration changes

**Alert Configuration**:
```yaml
# prometheus/alerts.yml
groups:
  - name: security
    rules:
      - alert: MultipleFailedLogins
        expr: rate(auth_failures_total[5m]) > 5
        annotations:
          summary: "Multiple failed login attempts detected"
      
      - alert: UnauthorizedAccess
        expr: rate(authorization_denied_total[5m]) > 0
        annotations:
          summary: "Unauthorized access attempt detected"
```

## Contact Information

**Security Team**:
- Email: security-team@company.com
- Slack: #security-incidents
- On-call: [phone number]

**External Resources**:
- HIPAA Breach Notification: https://www.hhs.gov/hipaa/for-professionals/breach-notification/
- FBI Cyber Division: https://www.fbi.gov/investigate/cyber
- CISA: https://www.cisa.gov/report
