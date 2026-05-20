#!/usr/bin/env python3
"""
Intrusion Detection System (IDS)

Lightweight IDS for detecting suspicious activity patterns via log analysis
and anomaly detection. Monitors security events, network patterns, and
system behavior for potential intrusions.
"""

import logging
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# IDS Rules and Signatures
# ============================================================================


class IDSRule:
    """IDS detection rule."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        severity: str,
        threshold: int = 1,
        time_window_seconds: int = 60,
        pattern: Optional[str] = None,
    ):
        """Initialize IDS rule.

        Args:
            rule_id: Unique rule identifier
            name: Human-readable rule name
            severity: Rule severity (low, medium, high, critical)
            threshold: Number of events to trigger alert
            time_window_seconds: Time window for threshold counting
            pattern: Optional regex pattern to match
        """
        self.rule_id = rule_id
        self.name = name
        self.severity = severity
        self.threshold = threshold
        self.time_window_seconds = time_window_seconds
        self.pattern = re.compile(pattern) if pattern else None


def get_ids_rules() -> List[IDSRule]:
    """Get IDS detection rules.

    Returns:
        List of IDS rules
    """
    return [
        # Brute Force Attacks
        IDSRule(
            rule_id="IDS-001",
            name="Brute Force Login Attempt",
            severity="high",
            threshold=5,
            time_window_seconds=60,
        ),
        IDSRule(
            rule_id="IDS-002",
            name="Distributed Brute Force Attack",
            severity="critical",
            threshold=10,
            time_window_seconds=300,
        ),
        # Unauthorized Access
        IDSRule(
            rule_id="IDS-003",
            name="Repeated Unauthorized Access Attempts",
            severity="high",
            threshold=3,
            time_window_seconds=60,
        ),
        IDSRule(
            rule_id="IDS-004",
            name="Privilege Escalation Attempt",
            severity="critical",
            threshold=1,
            time_window_seconds=60,
        ),
        # Data Exfiltration
        IDSRule(
            rule_id="IDS-005",
            name="Suspicious Data Download Volume",
            severity="high",
            threshold=100,  # 100 downloads
            time_window_seconds=300,
        ),
        IDSRule(
            rule_id="IDS-006",
            name="Bulk Data Access Pattern",
            severity="medium",
            threshold=50,
            time_window_seconds=60,
        ),
        # Scanning and Reconnaissance
        IDSRule(
            rule_id="IDS-007",
            name="Port Scanning Detected",
            severity="medium",
            threshold=10,
            time_window_seconds=60,
        ),
        IDSRule(
            rule_id="IDS-008",
            name="Directory Traversal Attempt",
            severity="high",
            threshold=3,
            time_window_seconds=60,
        ),
        # Malicious Payloads
        IDSRule(
            rule_id="IDS-009",
            name="SQL Injection Attempt",
            severity="critical",
            threshold=1,
            time_window_seconds=60,
        ),
        IDSRule(
            rule_id="IDS-010",
            name="XSS Attack Attempt",
            severity="high",
            threshold=1,
            time_window_seconds=60,
        ),
        # Anomalous Behavior
        IDSRule(
            rule_id="IDS-011",
            name="Unusual Access Time",
            severity="low",
            threshold=1,
            time_window_seconds=60,
        ),
        IDSRule(
            rule_id="IDS-012",
            name="Geolocation Anomaly",
            severity="medium",
            threshold=1,
            time_window_seconds=60,
        ),
        # System Compromise
        IDSRule(
            rule_id="IDS-013",
            name="Malware Upload Detected",
            severity="critical",
            threshold=1,
            time_window_seconds=60,
        ),
        IDSRule(
            rule_id="IDS-014",
            name="Suspicious Process Execution",
            severity="high",
            threshold=1,
            time_window_seconds=60,
        ),
    ]


# ============================================================================
# IDS Event Types
# ============================================================================


class IDSEvent:
    """IDS security event."""

    def __init__(
        self,
        event_type: str,
        source_ip: str,
        username: Optional[str] = None,
        details: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Initialize IDS event.

        Args:
            event_type: Type of security event
            source_ip: Source IP address
            username: Username if available
            details: Additional event details
            timestamp: Event timestamp
        """
        self.event_type = event_type
        self.source_ip = source_ip
        self.username = username
        self.details = details
        self.timestamp = timestamp or datetime.utcnow()


# ============================================================================
# IDS Engine
# ============================================================================


class IDSEngine:
    """Intrusion Detection System engine."""

    def __init__(self, rules: Optional[List[IDSRule]] = None):
        """Initialize IDS engine.

        Args:
            rules: List of IDS rules (defaults to standard rules)
        """
        self.rules = rules or get_ids_rules()

        # Event tracking
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.event_counts: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))

        # Alert tracking
        self.alerts: List[Dict] = []
        self.alerted_ips: Dict[str, datetime] = {}  # IP -> last alert time
        self.alert_cooldown_seconds = 300  # 5 minutes between alerts per IP

        # Baseline tracking for anomaly detection
        self.baseline_access_times: Dict[str, List[int]] = defaultdict(list)  # username -> hours
        self.baseline_locations: Dict[str, Set[str]] = defaultdict(set)  # username -> IPs

        logger.info(f"IDS engine initialized with {len(self.rules)} rules")

    def process_event(self, event: IDSEvent) -> Optional[Dict]:
        """Process security event through IDS rules.

        Args:
            event: IDS event to process

        Returns:
            Alert dict if rule triggered, None otherwise
        """
        # Store event
        self.events.append(event)

        # Update baseline data
        self._update_baseline(event)

        # Check each rule
        for rule in self.rules:
            if self._check_rule(rule, event):
                alert = self._create_alert(rule, event)

                # Check alert cooldown
                if self._should_alert(event.source_ip):
                    self.alerts.append(alert)
                    self.alerted_ips[event.source_ip] = datetime.utcnow()
                    logger.warning(f"IDS ALERT: {alert}")
                    return alert

        return None

    def _check_rule(self, rule: IDSRule, event: IDSEvent) -> bool:
        """Check if event triggers rule.

        Args:
            rule: IDS rule to check
            event: Event to check against rule

        Returns:
            True if rule triggered, False otherwise
        """
        # Map event types to rules
        event_rule_mapping = {
            "login_failed": ["IDS-001", "IDS-002"],
            "unauthorized_access_attempt": ["IDS-003"],
            "privilege_escalation_attempt": ["IDS-004"],
            "file_download": ["IDS-005", "IDS-006"],
            "port_scan": ["IDS-007"],
            "path_traversal": ["IDS-008"],
            "sql_injection": ["IDS-009"],
            "xss_attempt": ["IDS-010"],
            "unusual_access_time": ["IDS-011"],
            "geolocation_anomaly": ["IDS-012"],
            "malware_detected": ["IDS-013"],
            "suspicious_process": ["IDS-014"],
        }

        # Check if rule applies to this event type
        if rule.rule_id not in event_rule_mapping.get(event.event_type, []):
            return False

        # Track event for threshold counting
        key = f"{rule.rule_id}:{event.source_ip}"
        self.event_counts[rule.rule_id][event.source_ip].append(event.timestamp)

        # Clean old events outside time window
        cutoff_time = datetime.utcnow() - timedelta(seconds=rule.time_window_seconds)
        event_queue = self.event_counts[rule.rule_id][event.source_ip]
        while event_queue and event_queue[0] < cutoff_time:
            event_queue.popleft()

        # Check threshold
        if len(event_queue) >= rule.threshold:
            return True

        return False

    def _create_alert(self, rule: IDSRule, event: IDSEvent) -> Dict:
        """Create alert from triggered rule.

        Args:
            rule: Triggered rule
            event: Event that triggered rule

        Returns:
            Alert dictionary
        """
        return {
            "alert_id": f"ALERT-{len(self.alerts) + 1}",
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "severity": rule.severity,
            "source_ip": event.source_ip,
            "username": event.username,
            "event_type": event.event_type,
            "details": event.details,
            "timestamp": datetime.utcnow().isoformat(),
            "event_count": len(self.event_counts[rule.rule_id][event.source_ip]),
        }

    def _should_alert(self, ip: str) -> bool:
        """Check if alert should be sent (cooldown check).

        Args:
            ip: Source IP address

        Returns:
            True if should alert, False if in cooldown
        """
        if ip not in self.alerted_ips:
            return True

        last_alert = self.alerted_ips[ip]
        elapsed = (datetime.utcnow() - last_alert).total_seconds()

        return elapsed >= self.alert_cooldown_seconds

    def _update_baseline(self, event: IDSEvent) -> None:
        """Update baseline data for anomaly detection.

        Args:
            event: Event to update baseline with
        """
        if not event.username:
            return

        # Track access times (hour of day)
        hour = event.timestamp.hour
        if hour not in self.baseline_access_times[event.username]:
            self.baseline_access_times[event.username].append(hour)

        # Track locations (IP addresses)
        self.baseline_locations[event.username].add(event.source_ip)

    def detect_anomaly(self, event: IDSEvent) -> Optional[str]:
        """Detect anomalous behavior based on baseline.

        Args:
            event: Event to check for anomalies

        Returns:
            Anomaly description if detected, None otherwise
        """
        if not event.username:
            return None

        anomalies = []

        # Check unusual access time
        hour = event.timestamp.hour
        baseline_hours = self.baseline_access_times.get(event.username, [])
        if baseline_hours and hour not in baseline_hours:
            # Allow some tolerance (±2 hours)
            nearby_hours = [(hour + i) % 24 for i in range(-2, 3)]
            if not any(h in baseline_hours for h in nearby_hours):
                anomalies.append(f"Unusual access time: {hour}:00 (baseline: {baseline_hours})")

        # Check geolocation anomaly (new IP)
        baseline_ips = self.baseline_locations.get(event.username, set())
        if baseline_ips and event.source_ip not in baseline_ips:
            # Check if IP is from same subnet (first 3 octets)
            ip_prefix = ".".join(event.source_ip.split(".")[:3])
            known_prefixes = {".".join(ip.split(".")[:3]) for ip in baseline_ips}
            if ip_prefix not in known_prefixes:
                anomalies.append(
                    f"New IP address: {event.source_ip} (baseline: {len(baseline_ips)} IPs)"
                )

        return "; ".join(anomalies) if anomalies else None

    def get_alerts(
        self,
        severity: Optional[str] = None,
        source_ip: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get recent alerts.

        Args:
            severity: Filter by severity
            source_ip: Filter by source IP
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        filtered_alerts = self.alerts

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity]

        if source_ip:
            filtered_alerts = [a for a in filtered_alerts if a["source_ip"] == source_ip]

        return filtered_alerts[-limit:]

    def get_statistics(self) -> Dict:
        """Get IDS statistics.

        Returns:
            Statistics dictionary
        """
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity_counts[alert["severity"]] += 1

        return {
            "total_events": len(self.events),
            "total_alerts": len(self.alerts),
            "alerts_by_severity": dict(severity_counts),
            "unique_source_ips": len(set(e.source_ip for e in self.events)),
            "rules_active": len(self.rules),
        }


# ============================================================================
# Integration with Security Logging
# ============================================================================


def create_ids_event_from_security_log(
    event_type: str,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[str] = None,
) -> IDSEvent:
    """Create IDS event from security log entry.

    Args:
        event_type: Security event type
        username: Username if available
        ip_address: Source IP address
        details: Event details

    Returns:
        IDS event
    """
    return IDSEvent(
        event_type=event_type,
        source_ip=ip_address or "unknown",
        username=username,
        details=details,
    )


# Global IDS engine instance
_ids_engine: Optional[IDSEngine] = None


def get_ids_engine() -> IDSEngine:
    """Get global IDS engine instance.

    Returns:
        IDS engine
    """
    global _ids_engine
    if _ids_engine is None:
        _ids_engine = IDSEngine()
    return _ids_engine


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Example: Test IDS engine
    logging.basicConfig(level=logging.INFO)

    ids = IDSEngine()

    # Simulate brute force attack
    for i in range(6):
        event = IDSEvent(
            event_type="login_failed",
            source_ip="192.168.1.100",
            username="admin",
            details="Invalid password",
        )
        alert = ids.process_event(event)
        if alert:
            print(f"ALERT: {alert['rule_name']} - {alert['severity']}")

    # Simulate SQL injection
    event = IDSEvent(
        event_type="sql_injection",
        source_ip="10.0.0.50",
        username="attacker",
        details="UNION SELECT * FROM users",
    )
    alert = ids.process_event(event)
    if alert:
        print(f"ALERT: {alert['rule_name']} - {alert['severity']}")

    # Get statistics
    stats = ids.get_statistics()
    print(f"\nIDS Statistics: {stats}")
