# Dependency Vulnerability Scanning

## Overview

Third-party dependencies are a common attack vector. This guide covers automated dependency vulnerability scanning for the platform.

## Tools

### 1. Safety

Checks Python dependencies against known vulnerability databases.

#### Installation

```bash
pip install safety
```

#### Usage

```bash
# Scan installed packages
safety check

# Scan requirements file
safety check -r requirements.txt

# JSON output for CI/CD
safety check --json --output safety-report.json

# Fail on any vulnerability
safety check --exit-code
```

#### CI/CD Integration

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request]

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install safety
      
      - name: Run Safety check
        run: safety check --exit-code
```

### 2. pip-audit

Official PyPA tool for auditing Python packages.

#### Installation

```bash
pip install pip-audit
```

#### Usage

```bash
# Audit installed packages
pip-audit

# Audit requirements file
pip-audit -r requirements.txt

# JSON output
pip-audit --format json --output audit-report.json

# Fix vulnerabilities automatically
pip-audit --fix
```

### 3. Dependabot

GitHub's automated dependency updates.

#### Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    
    # Group minor and patch updates
    groups:
      development-dependencies:
        dependency-type: "development"
      production-dependencies:
        dependency-type: "production"
    
    # Ignore specific dependencies
    ignore:
      - dependency-name: "torch"
        update-types: ["version-update:semver-major"]
  
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 4. Snyk

Commercial tool with free tier for open source.

#### Installation

```bash
npm install -g snyk
snyk auth
```

#### Usage

```bash
# Test for vulnerabilities
snyk test

# Monitor project
snyk monitor

# Fix vulnerabilities
snyk fix
```

## the platform Integration

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: safety-check
        name: Safety vulnerability scan
        entry: safety check
        language: system
        pass_filenames: false
        always_run: true
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install safety pip-audit
      
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true
          python scripts/check_safety_results.py safety-report.json
      
      - name: Run pip-audit
        run: |
          pip-audit --format json --output audit-report.json || true
          python scripts/check_audit_results.py audit-report.json
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            audit-report.json
```

### Result Checker Script

Create `scripts/check_safety_results.py`:

```python
#!/usr/bin/env python3
"""Check Safety scan results and fail on HIGH/CRITICAL vulnerabilities."""

import json
import sys
from pathlib import Path

def check_safety_results(report_path: Path) -> int:
    """
    Check Safety scan results.
    
    Returns:
        0 if no HIGH/CRITICAL vulnerabilities
        1 if HIGH/CRITICAL vulnerabilities found
    """
    with open(report_path) as f:
        data = json.load(f)
    
    vulnerabilities = data.get('vulnerabilities', [])
    
    high_critical = [
        v for v in vulnerabilities
        if v.get('severity', '').upper() in ['HIGH', 'CRITICAL']
    ]
    
    if high_critical:
        print(f"❌ Found {len(high_critical)} HIGH/CRITICAL vulnerabilities:")
        for vuln in high_critical:
            print(f"  - {vuln['package']}: {vuln['vulnerability']}")
            print(f"    Severity: {vuln['severity']}")
            print(f"    Affected: {vuln['affected_versions']}")
            print(f"    Fixed in: {vuln.get('fixed_versions', 'N/A')}")
        return 1
    
    print(f"✅ No HIGH/CRITICAL vulnerabilities found")
    print(f"   Total vulnerabilities: {len(vulnerabilities)}")
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: check_safety_results.py <report.json>")
        sys.exit(1)
    
    sys.exit(check_safety_results(Path(sys.argv[1])))
```

## Vulnerability Response Process

### 1. Detection

- Automated scans run on every PR
- Weekly scheduled scans
- Dependabot alerts

### 2. Triage

```python
# Severity classification
CRITICAL = "Immediate action required (< 24 hours)"
HIGH = "Action required (< 7 days)"
MEDIUM = "Action required (< 30 days)"
LOW = "Action required (< 90 days)"
```

### 3. Remediation

```bash
# Update specific package
pip install --upgrade package-name==fixed-version

# Update all packages (careful!)
pip install --upgrade -r requirements.txt

# Test after update
pytest tests/
```

### 4. Documentation

Document in `SECURITY.md`:

```markdown
## Known Vulnerabilities

### CVE-2024-XXXXX (RESOLVED)
- **Package:** requests
- **Severity:** HIGH
- **Affected:** 2.28.0 - 2.28.2
- **Fixed in:** 2.29.0
- **Resolution:** Updated to 2.29.0 on 2024-01-15
- **Verification:** All tests passing
```

## Pinning Strategy

### Production

```txt
# requirements.txt - Pin exact versions
torch==2.1.0
numpy==1.24.3
pandas==2.0.3
```

### Development

```txt
# requirements-dev.txt - Allow minor updates
pytest>=7.3.0,<8.0.0
black>=23.0.0,<24.0.0
```

### Constraints File

```txt
# constraints.txt - Upper bounds for compatibility
torch<3.0.0
numpy<2.0.0
pandas<3.0.0
```

## Monitoring

### GitHub Security Advisories

Enable in repository settings:
- Settings → Security → Dependabot alerts
- Settings → Security → Dependabot security updates

### Email Notifications

Configure in `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "security-team"
    assignees:
      - "lead-developer"
```

### Slack Integration

```python
# scripts/notify_vulnerabilities.py
import requests
import json

def notify_slack(webhook_url: str, vulnerabilities: list):
    """Send vulnerability notification to Slack."""
    message = {
        "text": f"🚨 {len(vulnerabilities)} vulnerabilities found",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Security Alert*\n{len(vulnerabilities)} vulnerabilities detected"
                }
            }
        ]
    }
    
    for vuln in vulnerabilities[:5]:  # First 5
        message["blocks"].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{vuln['package']}*: {vuln['vulnerability']}\nSeverity: {vuln['severity']}"
            }
        })
    
    requests.post(webhook_url, json=message)
```

## Best Practices

1. **Scan frequently**: Daily in CI/CD, weekly scheduled
2. **Pin versions**: Use exact versions in production
3. **Update regularly**: Don't let dependencies get too old
4. **Test updates**: Run full test suite after updates
5. **Monitor advisories**: Subscribe to security mailing lists
6. **Document exceptions**: If you can't update, document why
7. **Use constraints**: Prevent incompatible updates

## Common Issues

### Issue 1: False Positives

**Solution:** Use ignore files

```yaml
# .safety-policy.yml
security:
  ignore-vulnerabilities:
    - id: 12345
      reason: "Not applicable - I don't use affected feature"
      expires: "2024-12-31"
```

### Issue 2: Transitive Dependencies

**Solution:** Use `pip-audit --fix` or update parent package

```bash
# Find which package depends on vulnerable package
pip show package-name

# Update parent package
pip install --upgrade parent-package
```

### Issue 3: No Fix Available

**Solution:** Document risk and implement mitigations

```markdown
## Accepted Risks

### CVE-2024-XXXXX
- **Package:** old-library
- **Severity:** MEDIUM
- **Status:** No fix available
- **Mitigation:** Input validation, sandboxing
- **Review date:** 2024-06-01
```

## References

- [Safety Documentation](https://pyup.io/safety/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [NIST NVD](https://nvd.nist.gov/)
- [CVE Database](https://cve.mitre.org/)
