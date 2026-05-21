# Logging Sanitization Guide

## Overview

Proper log sanitization prevents sensitive data leakage, injection attacks, and compliance violations. This guide covers secure logging practices for the platform.

## Sensitive Data Categories

### Never Log These

1. **Credentials**
   - Passwords (plaintext or hashed)
   - API keys and tokens
   - Private keys and certificates
   - Session IDs

2. **Personal Health Information (PHI)**
   - Patient names
   - Medical record numbers (MRNs)
   - Social security numbers
   - Dates of birth
   - Addresses and phone numbers

3. **Financial Data**
   - Credit card numbers
   - Bank account numbers
   - Payment tokens

4. **Internal Secrets**
   - Database connection strings with passwords
   - Encryption keys
   - Internal IP addresses (in production)

## Sanitization Patterns

### Pattern 1: Redaction

```python
import re
import logging

def sanitize_log_message(message: str) -> str:
    """Sanitize log message by redacting sensitive patterns."""
    
    # Redact email addresses
    message = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL_REDACTED]',
        message
    )
    
    # Redact credit card numbers
    message = re.sub(
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        '[CARD_REDACTED]',
        message
    )
    
    # Redact API keys (common patterns)
    message = re.sub(
        r'\b[A-Za-z0-9]{32,}\b',
        '[KEY_REDACTED]',
        message
    )
    
    # Redact IP addresses
    message = re.sub(
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        '[IP_REDACTED]',
        message
    )
    
    return message

# Usage
logger = logging.getLogger(__name__)
message = "User john@example.com logged in from 192.168.1.100"
logger.info(sanitize_log_message(message))
# Output: "User [EMAIL_REDACTED] logged in from [IP_REDACTED]"
```

### Pattern 2: Structured Logging with Filtering

```python
import logging
import json
from typing import Any, Dict

class SanitizingFormatter(logging.Formatter):
    """Custom formatter that sanitizes sensitive fields."""
    
    SENSITIVE_KEYS = {
        'password', 'token', 'api_key', 'secret', 'ssn',
        'credit_card', 'mrn', 'patient_name'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Sanitize extra fields
        if hasattr(record, 'extra'):
            record.extra = self._sanitize_dict(record.extra)
        
        # Sanitize message
        record.msg = sanitize_log_message(str(record.msg))
        
        return super().format(record)
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary."""
        sanitized = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_KEYS:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

# Setup
handler = logging.StreamHandler()
handler.setFormatter(SanitizingFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger(__name__)
logger.addHandler(handler)

# Usage
logger.info("User login", extra={
    'username': 'john_doe',
    'password': 'secret123',  # Will be redacted
    'ip_address': '192.168.1.100'
})
```

### Pattern 3: Partial Masking

```python
def mask_sensitive_data(value: str, visible_chars: int = 4) -> str:
    """Mask sensitive data, showing only last N characters."""
    if len(value) <= visible_chars:
        return '*' * len(value)
    return '*' * (len(value) - visible_chars) + value[-visible_chars:]

# Usage
logger.info(f"Processing card: {mask_sensitive_data('4532123456789012', 4)}")
# Output: "Processing card: ************9012"

logger.info(f"API key: {mask_sensitive_data('sk_live_abc123def456', 4)}")
# Output: "API key: **********f456"
```

## the platform Integration

### Security Audit Trail

The `SecurityAuditTrail` class automatically sanitizes logs:

```python
from src.security.audit_trail import SecurityAuditTrail

audit = SecurityAuditTrail()

# Automatically sanitizes sensitive fields
audit.log_policy_applied(
    policy_name="network_binding",
    decision="allowed",
    context={
        'host': '127.0.0.1',
        'port': 8000,
        'api_key': 'secret_key_123'  # Automatically redacted
    }
)
```

### Custom Logger Configuration

```python
import logging
from src.security.logging_sanitizer import SanitizingFormatter

def setup_secure_logging():
    """Configure secure logging for the platform."""
    
    # Create sanitizing formatter
    formatter = SanitizingFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler for audit logs
    audit_handler = logging.FileHandler('logs/audit.log')
    audit_handler.setFormatter(formatter)
    audit_handler.setLevel(logging.INFO)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(audit_handler)
    root_logger.addHandler(console_handler)
```

## Environment-Specific Logging

### Production

```python
import os

if os.getenv('ENVIRONMENT') == 'production':
    # Strict sanitization
    logging.getLogger().setLevel(logging.WARNING)
    
    # Disable debug logs
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    # Enable audit logging
    audit_logger = logging.getLogger('security.audit')
    audit_logger.setLevel(logging.INFO)
```

### Development

```python
if os.getenv('ENVIRONMENT') == 'development':
    # More verbose logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Still sanitize sensitive data
    # (use test data, not real PHI)
```

## Common Pitfalls

### ❌ Bad: Logging Raw User Input

```python
# NEVER DO THIS
logger.info(f"User search query: {user_input}")
# Risk: Log injection, sensitive data leakage
```

### ✅ Good: Sanitize Before Logging

```python
# DO THIS
logger.info(f"User search query: {sanitize_log_message(user_input)}")
```

### ❌ Bad: Logging Exception with Sensitive Data

```python
# NEVER DO THIS
try:
    authenticate(username, password)
except Exception as e:
    logger.error(f"Auth failed: {e}")  # May contain password
```

### ✅ Good: Log Exception Type Only

```python
# DO THIS
try:
    authenticate(username, password)
except Exception as e:
    logger.error(f"Auth failed for user {username}: {type(e).__name__}")
```

## Compliance

### HIPAA Requirements

- **§164.308(a)(1)(ii)(D)**: Log access to PHI
- **§164.312(b)**: Audit controls
- **§164.530(j)**: Accountability

**Implementation:**
```python
# Log access without PHI
logger.info(
    "Patient record accessed",
    extra={
        'user_id': user.id,
        'record_id': record.id,  # Use ID, not name
        'action': 'view',
        'timestamp': datetime.utcnow()
    }
)
```

### GDPR Requirements

- **Article 32**: Security of processing
- **Article 33**: Breach notification

**Implementation:**
```python
# Use pseudonymization
logger.info(
    "User action",
    extra={
        'user_hash': hashlib.sha256(user_id.encode()).hexdigest()[:16],
        'action': 'data_export',
        'timestamp': datetime.utcnow()
    }
)
```

## Testing

```python
import pytest
from src.security.logging_sanitizer import sanitize_log_message

def test_email_redaction():
    message = "User john@example.com logged in"
    sanitized = sanitize_log_message(message)
    assert "john@example.com" not in sanitized
    assert "[EMAIL_REDACTED]" in sanitized

def test_credit_card_redaction():
    message = "Payment with card 4532-1234-5678-9012"
    sanitized = sanitize_log_message(message)
    assert "4532-1234-5678-9012" not in sanitized
    assert "[CARD_REDACTED]" in sanitized

def test_api_key_redaction():
    message = "Using API key sk_live_abc123def456ghi789"
    sanitized = sanitize_log_message(message)
    assert "sk_live_abc123def456ghi789" not in sanitized
    assert "[KEY_REDACTED]" in sanitized
```

## References

- [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)
- [NIST SP 800-92 - Guide to Computer Security Log Management](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-92.pdf)
- [CWE-532: Insertion of Sensitive Information into Log File](https://cwe.mitre.org/data/definitions/532.html)
