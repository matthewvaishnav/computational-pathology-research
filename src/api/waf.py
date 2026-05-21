#!/usr/bin/env python3
"""
Web Application Firewall (WAF) Integration

Provides WAF middleware for FastAPI with ModSecurity-like rule engine
for detecting and blocking common web attacks (SQL injection, XSS, etc.)
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ============================================================================
# WAF Rule Engine
# ============================================================================


class WAFRule:
    """WAF rule definition."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        severity: str = "medium",
        action: str = "block",
        targets: Optional[List[str]] = None,
    ):
        """Initialize WAF rule.

        Args:
            rule_id: Unique rule identifier
            name: Human-readable rule name
            pattern: Regex pattern to match
            severity: Rule severity (low, medium, high, critical)
            action: Action to take (log, block)
            targets: Request parts to check (query, body, headers, cookies, uri)
        """
        self.rule_id = rule_id
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.severity = severity
        self.action = action
        self.targets = targets or ["query", "body", "headers", "uri"]

    def matches(self, value: str) -> bool:
        """Check if value matches rule pattern."""
        try:
            return bool(self.pattern.search(value))
        except Exception as e:
            logger.error(f"Rule {self.rule_id} pattern match error: {e}")
            return False


# ============================================================================
# OWASP Core Rule Set (CRS) - Simplified
# ============================================================================


def get_owasp_crs_rules() -> List[WAFRule]:
    """Get OWASP Core Rule Set (simplified version).

    Returns:
        List of WAF rules
    """
    return [
        # SQL Injection
        WAFRule(
            rule_id="942100",
            name="SQL Injection Attack Detected via libinjection",
            pattern=r"(\bunion\b.*\bselect\b|\bselect\b.*\bfrom\b|\binsert\b.*\binto\b|\bdelete\b.*\bfrom\b|\bdrop\b.*\btable\b)",
            severity="critical",
            action="block",
            targets=["query", "body"],
        ),
        WAFRule(
            rule_id="942110",
            name="SQL Injection Attack: Common Injection Testing Detected",
            pattern=r"(\'|\"|;|--|\#|\/\*|\*\/|xp_|sp_|exec\(|execute\()",
            severity="high",
            action="block",
            targets=["query", "body"],
        ),
        # XSS (Cross-Site Scripting)
        WAFRule(
            rule_id="941100",
            name="XSS Attack Detected via libinjection",
            pattern=r"(<script[^>]*>.*?</script>|javascript:|onerror=|onload=|<iframe|<object|<embed)",
            severity="high",
            action="block",
            targets=["query", "body", "headers"],
        ),
        WAFRule(
            rule_id="941110",
            name="XSS Filter - Category 1: Script Tag Vector",
            pattern=r"<script[^>]*>",
            severity="high",
            action="block",
            targets=["query", "body"],
        ),
        # Path Traversal
        WAFRule(
            rule_id="930100",
            name="Path Traversal Attack (/../)",
            pattern=r"(\.\./|\.\.\\|%2e%2e/|%2e%2e\\)",
            severity="high",
            action="block",
            targets=["query", "uri"],
        ),
        WAFRule(
            rule_id="930110",
            name="Path Traversal Attack (/.../)",
            pattern=r"(/\.\./|/\.\.\\|\\\.\.\\|\\\.\.\/)",
            severity="high",
            action="block",
            targets=["query", "uri"],
        ),
        # Remote File Inclusion (RFI)
        WAFRule(
            rule_id="931100",
            name="Possible Remote File Inclusion (RFI) Attack: URL Parameter using IP Address",
            pattern=r"(http://|https://|ftp://|file://|php://|data://|expect://)",
            severity="high",
            action="block",
            targets=["query", "body"],
        ),
        # Command Injection
        WAFRule(
            rule_id="932100",
            name="Remote Command Execution: Unix Command Injection",
            pattern=r"(;|\||`|\$\(|&&|\|\||>|<|cat\s|ls\s|wget\s|curl\s|nc\s|bash\s|sh\s)",
            severity="critical",
            action="block",
            targets=["query", "body"],
        ),
        # LDAP Injection
        WAFRule(
            rule_id="950100",
            name="LDAP Injection Attack",
            pattern=r"(\(|\)|\*|\||&)",
            severity="medium",
            action="log",
            targets=["query", "body"],
        ),
        # XML External Entity (XXE)
        WAFRule(
            rule_id="960100",
            name="XML External Entity (XXE) Attack",
            pattern=r"(<!ENTITY|<!DOCTYPE|SYSTEM|PUBLIC)",
            severity="high",
            action="block",
            targets=["body"],
        ),
        # Server-Side Request Forgery (SSRF)
        WAFRule(
            rule_id="970100",
            name="Possible SSRF Attack: Internal IP Address",
            pattern=r"(127\.0\.0\.1|localhost|0\.0\.0\.0|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2[0-9]|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)",
            severity="high",
            action="block",
            targets=["query", "body"],
        ),
        # HTTP Protocol Violations
        WAFRule(
            rule_id="920100",
            name="Invalid HTTP Request Line",
            pattern=r"(GET\s+.*\s+HTTP/[0-9]\.[0-9].*GET|POST\s+.*\s+HTTP/[0-9]\.[0-9].*POST)",
            severity="medium",
            action="block",
            targets=["uri"],
        ),
        # Scanner Detection
        WAFRule(
            rule_id="913100",
            name="Security Scanner Detected",
            pattern=r"(nikto|nmap|sqlmap|burp|acunetix|nessus|openvas|metasploit|w3af)",
            severity="medium",
            action="block",
            targets=["headers"],
        ),
    ]


# ============================================================================
# WAF Engine
# ============================================================================


class WAFEngine:
    """Web Application Firewall engine."""

    def __init__(self, rules: Optional[List[WAFRule]] = None):
        """Initialize WAF engine.

        Args:
            rules: List of WAF rules (defaults to OWASP CRS)
        """
        self.rules = rules or get_owasp_crs_rules()
        self.blocked_ips: Dict[str, datetime] = {}  # IP -> block_until
        self.violation_counts: Dict[str, int] = {}  # IP -> count
        self.block_threshold = 5  # Block after N violations
        self.block_duration_minutes = 60  # Block for 1 hour

        logger.info(f"WAF engine initialized with {len(self.rules)} rules")

    def check_request(self, request: Request) -> Optional[Dict]:
        """Check request against WAF rules.

        Args:
            request: FastAPI request

        Returns:
            Violation dict if rule matched, None otherwise
        """
        # Check if IP is blocked
        client_ip = request.client.host
        if self._is_ip_blocked(client_ip):
            return {
                "rule_id": "IP_BLOCKED",
                "name": "IP Address Blocked",
                "severity": "critical",
                "action": "block",
                "message": "Your IP address has been temporarily blocked due to suspicious activity",
            }

        # Extract request data
        request_data = self._extract_request_data(request)

        # Check each rule
        for rule in self.rules:
            for target, value in request_data.items():
                if target in rule.targets and rule.matches(value):
                    violation = {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "severity": rule.severity,
                        "action": rule.action,
                        "target": target,
                        "matched_value": value[:100],  # Truncate for logging
                        "client_ip": client_ip,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    # Log violation
                    logger.warning(f"WAF violation: {violation}")

                    # Track violations per IP
                    self._record_violation(client_ip)

                    # Return violation if action is block
                    if rule.action == "block":
                        return violation

        return None

    def _extract_request_data(self, request: Request) -> Dict[str, str]:
        """Extract data from request for rule checking.

        Args:
            request: FastAPI request

        Returns:
            Dict of target -> value
        """
        data = {}

        # URI
        data["uri"] = str(request.url.path)

        # Query parameters
        if request.url.query:
            data["query"] = request.url.query

        # Headers
        headers_str = " ".join([f"{k}:{v}" for k, v in request.headers.items()])
        data["headers"] = headers_str

        # Cookies
        if request.cookies:
            cookies_str = " ".join([f"{k}={v}" for k, v in request.cookies.items()])
            data["cookies"] = cookies_str

        # Body (will be populated by middleware)
        # Note: Body reading is async and can only be done once
        # Middleware should cache body for WAF inspection

        return data

    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked.

        Args:
            ip: Client IP address

        Returns:
            True if blocked, False otherwise
        """
        if ip in self.blocked_ips:
            block_until = self.blocked_ips[ip]
            if datetime.utcnow() < block_until:
                return True
            else:
                # Block expired, remove
                del self.blocked_ips[ip]
                if ip in self.violation_counts:
                    del self.violation_counts[ip]

        return False

    def _record_violation(self, ip: str) -> None:
        """Record violation for IP and block if threshold exceeded.

        Args:
            ip: Client IP address
        """
        # Increment violation count
        self.violation_counts[ip] = self.violation_counts.get(ip, 0) + 1

        # Check if threshold exceeded
        if self.violation_counts[ip] >= self.block_threshold:
            block_until = datetime.utcnow() + timedelta(minutes=self.block_duration_minutes)
            self.blocked_ips[ip] = block_until
            logger.warning(
                f"IP {ip} blocked until {block_until} after {self.violation_counts[ip]} violations"
            )


# ============================================================================
# FastAPI Middleware
# ============================================================================


class WAFMiddleware:
    """FastAPI middleware for WAF integration."""

    def __init__(self, app, waf_engine: Optional[WAFEngine] = None):
        """Initialize WAF middleware.

        Args:
            app: FastAPI app
            waf_engine: WAF engine instance
        """
        self.app = app
        self.waf_engine = waf_engine or WAFEngine()

    async def __call__(self, request: Request, call_next):
        """Process request through WAF.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Check request against WAF rules
        violation = self.waf_engine.check_request(request)

        if violation:
            # Block request
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Request blocked by Web Application Firewall",
                    "rule_id": violation["rule_id"],
                    "message": violation.get(
                        "message", "Your request was blocked due to security policy"
                    ),
                },
            )

        # Continue to next handler
        response = await call_next(request)

        return response


# ============================================================================
# Convenience Functions
# ============================================================================


def create_waf_middleware(app, custom_rules: Optional[List[WAFRule]] = None):
    """Create WAF middleware for FastAPI app.

    Args:
        app: FastAPI app
        custom_rules: Optional custom rules (in addition to OWASP CRS)

    Returns:
        WAF middleware
    """
    rules = get_owasp_crs_rules()
    if custom_rules:
        rules.extend(custom_rules)

    waf_engine = WAFEngine(rules=rules)
    return WAFMiddleware(app, waf_engine=waf_engine)


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Example: Test WAF rules
    logging.basicConfig(level=logging.INFO)

    # Create WAF engine
    waf = WAFEngine()

    # Test SQL injection
    test_queries = [
        "id=1 UNION SELECT * FROM users",
        "name=admin' OR '1'='1",
        "search=<script>alert('xss')</script>",
        "file=../../../../etc/passwd",
        "url=http://127.0.0.1:8080/admin",
    ]

    for query in test_queries:
        # Simulate request
        class MockRequest:
            def __init__(self, query_string):
                self.client = type("obj", (object,), {"host": "192.168.1.100"})
                self.url = type("obj", (object,), {"path": "/test", "query": query_string})
                self.headers = {}
                self.cookies = {}

        request = MockRequest(query)
        violation = waf.check_request(request)

        if violation:
            logger.info(f"✗ BLOCKED: {query}")
            logger.info(f"  Rule: {violation['rule_id']} - {violation['name']}")
        else:
            logger.info(f"✓ ALLOWED: {query}")
