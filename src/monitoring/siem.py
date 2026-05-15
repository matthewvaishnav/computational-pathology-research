#!/usr/bin/env python3
"""
Security Information and Event Management (SIEM)

Lightweight SIEM for aggregating, correlating, and analyzing security events
from multiple sources (logs, IDS, WAF, audit trails).
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# SIEM Event Correlation
# ============================================================================


class SIEMEvent:
    """SIEM security event."""

    def __init__(
        self,
        source: str,
        event_type: str,
        severity: str,
        source_ip: str,
        username: Optional[str] = None,
        details: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Initialize SIEM event.

        Args:
            source: Event source (ids, waf, audit, api)
            event_type: Type of security event
            severity: Event severity (low, medium, high, critical)
            source_ip: Source IP address
            username: Username if available
            details: Event details
            timestamp: Event timestamp
        """
        self.source = source
        self.event_type = event_type
        self.severity = severity
        self.source_ip = source_ip
        self.username = username
        self.details = details
        self.timestamp = timestamp or datetime.utcnow()


class CorrelationRule:
    """SIEM correlation rule for detecting attack patterns."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        event_sequence: List[str],
        time_window_seconds: int = 300,
        severity: str = "high",
    ):
        """Initialize correlation rule.

        Args:
            rule_id: Unique rule identifier
            name: Rule name
            description: Rule description
            event_sequence: Sequence of event types to match
            time_window_seconds: Time window for correlation
            severity: Incident severity if rule matches
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.event_sequence = event_sequence
        self.time_window_seconds = time_window_seconds
        self.severity = severity


def get_correlation_rules() -> List[CorrelationRule]:
    """Get SIEM correlation rules.

    Returns:
        List of correlation rules
    """
    return [
        CorrelationRule(
            rule_id="CORR-001",
            name="Multi-Stage Attack Pattern",
            description="Reconnaissance followed by exploitation attempt",
            event_sequence=["port_scan", "sql_injection"],
            time_window_seconds=600,
            severity="critical",
        ),
        CorrelationRule(
            rule_id="CORR-002",
            name="Credential Stuffing Attack",
            description="Multiple failed logins followed by successful login",
            event_sequence=["login_failed", "login_failed", "login_success"],
            time_window_seconds=300,
            severity="high",
        ),
        CorrelationRule(
            rule_id="CORR-003",
            name="Data Exfiltration Pattern",
            description="Unauthorized access followed by bulk downloads",
            event_sequence=["unauthorized_access_attempt", "file_download"],
            time_window_seconds=600,
            severity="critical",
        ),
        CorrelationRule(
            rule_id="CORR-004",
            name="Privilege Escalation Chain",
            description="Failed access followed by privilege escalation",
            event_sequence=["unauthorized_access_attempt", "privilege_escalation_attempt"],
            time_window_seconds=300,
            severity="critical",
        ),
        CorrelationRule(
            rule_id="CORR-005",
            name="Lateral Movement",
            description="Successful login from multiple IPs in short time",
            event_sequence=["login_success", "login_success"],
            time_window_seconds=60,
            severity="high",
        ),
    ]


# ============================================================================
# SIEM Engine
# ============================================================================


class SIEMEngine:
    """Security Information and Event Management engine."""

    def __init__(self, correlation_rules: Optional[List[CorrelationRule]] = None):
        """Initialize SIEM engine.

        Args:
            correlation_rules: List of correlation rules
        """
        self.correlation_rules = correlation_rules or get_correlation_rules()

        # Event storage
        self.events: deque = deque(maxlen=50000)  # Keep last 50k events
        self.events_by_ip: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.events_by_user: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Incident tracking
        self.incidents: List[Dict] = []

        # Statistics
        self.stats = {
            "events_by_source": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "events_by_type": defaultdict(int),
        }

        logger.info(f"SIEM engine initialized with {len(self.correlation_rules)} correlation rules")

    def ingest_event(self, event: SIEMEvent) -> Optional[Dict]:
        """Ingest security event into SIEM.

        Args:
            event: SIEM event to ingest

        Returns:
            Incident dict if correlation rule triggered, None otherwise
        """
        # Store event
        self.events.append(event)
        self.events_by_ip[event.source_ip].append(event)
        if event.username:
            self.events_by_user[event.username].append(event)

        # Update statistics
        self.stats["events_by_source"][event.source] += 1
        self.stats["events_by_severity"][event.severity] += 1
        self.stats["events_by_type"][event.event_type] += 1

        # Check correlation rules
        for rule in self.correlation_rules:
            incident = self._check_correlation(rule, event)
            if incident:
                self.incidents.append(incident)
                logger.critical(f"SIEM INCIDENT: {incident}")
                return incident

        return None

    def _check_correlation(self, rule: CorrelationRule, event: SIEMEvent) -> Optional[Dict]:
        """Check if event triggers correlation rule.

        Args:
            rule: Correlation rule to check
            event: Event to check

        Returns:
            Incident dict if rule triggered, None otherwise
        """
        # Get recent events from same IP
        ip_events = list(self.events_by_ip[event.source_ip])

        # Filter events within time window
        cutoff_time = datetime.utcnow() - timedelta(seconds=rule.time_window_seconds)
        recent_events = [e for e in ip_events if e.timestamp >= cutoff_time]

        # Check if event sequence matches
        if self._matches_sequence(rule.event_sequence, recent_events):
            return self._create_incident(rule, event, recent_events)

        return None

    def _matches_sequence(self, sequence: List[str], events: List[SIEMEvent]) -> bool:
        """Check if events match sequence pattern.

        Args:
            sequence: Event type sequence to match
            events: List of events to check

        Returns:
            True if sequence matches, False otherwise
        """
        if len(events) < len(sequence):
            return False

        # Check for sequence in events (order matters)
        seq_idx = 0
        for event in events:
            if event.event_type == sequence[seq_idx]:
                seq_idx += 1
                if seq_idx == len(sequence):
                    return True

        return False

    def _create_incident(
        self,
        rule: CorrelationRule,
        trigger_event: SIEMEvent,
        related_events: List[SIEMEvent],
    ) -> Dict:
        """Create security incident from triggered correlation rule.

        Args:
            rule: Triggered correlation rule
            trigger_event: Event that triggered the rule
            related_events: Related events in correlation

        Returns:
            Incident dictionary
        """
        return {
            "incident_id": f"INC-{len(self.incidents) + 1}",
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "description": rule.description,
            "severity": rule.severity,
            "source_ip": trigger_event.source_ip,
            "username": trigger_event.username,
            "timestamp": datetime.utcnow().isoformat(),
            "event_count": len(related_events),
            "event_sources": list(set(e.source for e in related_events)),
            "event_types": [e.event_type for e in related_events],
        }

    def get_incidents(
        self,
        severity: Optional[str] = None,
        source_ip: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get security incidents.

        Args:
            severity: Filter by severity
            source_ip: Filter by source IP
            limit: Maximum number of incidents

        Returns:
            List of incidents
        """
        filtered = self.incidents

        if severity:
            filtered = [i for i in filtered if i["severity"] == severity]

        if source_ip:
            filtered = [i for i in filtered if i["source_ip"] == source_ip]

        return filtered[-limit:]

    def get_dashboard_data(self) -> Dict:
        """Get SIEM dashboard data.

        Returns:
            Dashboard data dictionary
        """
        # Calculate time-based metrics
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_24h = now - timedelta(hours=24)

        events_last_hour = sum(1 for e in self.events if e.timestamp >= last_hour)
        events_last_24h = sum(1 for e in self.events if e.timestamp >= last_24h)

        incidents_last_hour = sum(
            1 for i in self.incidents if datetime.fromisoformat(i["timestamp"]) >= last_hour
        )
        incidents_last_24h = sum(
            1 for i in self.incidents if datetime.fromisoformat(i["timestamp"]) >= last_24h
        )

        # Top attackers
        ip_counts = defaultdict(int)
        for event in self.events:
            if event.severity in ["high", "critical"]:
                ip_counts[event.source_ip] += 1

        top_attackers = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_events": len(self.events),
            "total_incidents": len(self.incidents),
            "events_last_hour": events_last_hour,
            "events_last_24h": events_last_24h,
            "incidents_last_hour": incidents_last_hour,
            "incidents_last_24h": incidents_last_24h,
            "events_by_source": dict(self.stats["events_by_source"]),
            "events_by_severity": dict(self.stats["events_by_severity"]),
            "top_attackers": [{"ip": ip, "count": count} for ip, count in top_attackers],
        }

    def search_events(
        self,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        source_ip: Optional[str] = None,
        username: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Search security events.

        Args:
            source: Filter by event source
            event_type: Filter by event type
            severity: Filter by severity
            source_ip: Filter by source IP
            username: Filter by username
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results

        Returns:
            List of matching events
        """
        filtered = list(self.events)

        if source:
            filtered = [e for e in filtered if e.source == source]

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if severity:
            filtered = [e for e in filtered if e.severity == severity]

        if source_ip:
            filtered = [e for e in filtered if e.source_ip == source_ip]

        if username:
            filtered = [e for e in filtered if e.username == username]

        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        # Convert to dict
        results = []
        for event in filtered[-limit:]:
            results.append(
                {
                    "source": event.source,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "source_ip": event.source_ip,
                    "username": event.username,
                    "details": event.details,
                    "timestamp": event.timestamp.isoformat(),
                }
            )

        return results


# ============================================================================
# Integration Helpers
# ============================================================================


def create_siem_event_from_ids_alert(alert: Dict) -> SIEMEvent:
    """Create SIEM event from IDS alert.

    Args:
        alert: IDS alert dictionary

    Returns:
        SIEM event
    """
    return SIEMEvent(
        source="ids",
        event_type=alert["event_type"],
        severity=alert["severity"],
        source_ip=alert["source_ip"],
        username=alert.get("username"),
        details=f"IDS Alert: {alert['rule_name']}",
    )


def create_siem_event_from_waf_violation(violation: Dict) -> SIEMEvent:
    """Create SIEM event from WAF violation.

    Args:
        violation: WAF violation dictionary

    Returns:
        SIEM event
    """
    return SIEMEvent(
        source="waf",
        event_type=violation["rule_id"],
        severity=violation["severity"],
        source_ip=violation["client_ip"],
        details=f"WAF Violation: {violation['name']}",
    )


def create_siem_event_from_audit_log(log_entry: Dict) -> SIEMEvent:
    """Create SIEM event from audit log entry.

    Args:
        log_entry: Audit log entry dictionary

    Returns:
        SIEM event
    """
    # Map event types to severity
    severity_map = {
        "login_failed": "medium",
        "unauthorized_access_attempt": "high",
        "privilege_escalation_attempt": "critical",
        "file_upload_failed": "low",
        "malware_detected": "critical",
    }

    severity = severity_map.get(log_entry["event_type"], "low")

    return SIEMEvent(
        source="audit",
        event_type=log_entry["event_type"],
        severity=severity,
        source_ip=log_entry.get("ip_address", "unknown"),
        username=log_entry.get("username"),
        details=log_entry.get("details"),
    )


# Global SIEM engine instance
_siem_engine: Optional[SIEMEngine] = None


def get_siem_engine() -> SIEMEngine:
    """Get global SIEM engine instance.

    Returns:
        SIEM engine
    """
    global _siem_engine
    if _siem_engine is None:
        _siem_engine = SIEMEngine()
    return _siem_engine


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Example: Test SIEM engine
    logging.basicConfig(level=logging.INFO)

    siem = SIEMEngine()

    # Simulate multi-stage attack
    events = [
        SIEMEvent("ids", "port_scan", "medium", "10.0.0.50", details="Port scan detected"),
        SIEMEvent("waf", "sql_injection", "critical", "10.0.0.50", details="SQL injection attempt"),
    ]

    for event in events:
        incident = siem.ingest_event(event)
        if incident:
            print(f"INCIDENT: {incident['rule_name']} - {incident['severity']}")

    # Get dashboard data
    dashboard = siem.get_dashboard_data()
    print(f"\nSIEM Dashboard: {dashboard}")
