"""
Data models for HistoCore Project Optimization Analysis System.

Defines core data structures for analysis results, issues, and optimization plans.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import jsonschema
from jsonschema import validate, ValidationError


class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(str, Enum):
    """Task priority levels."""
    P0 = "P0"  # Critical
    P1 = "P1"  # High
    P2 = "P2"  # Medium
    P3 = "P3"  # Low


class Role(str, Enum):
    """Engineering roles for task assignment."""
    BACKEND = "backend"
    ML = "ml"
    DEVOPS = "devops"
    SECURITY = "security"
    QA = "qa"


@dataclass
class Issue:
    """Individual finding from analysis."""
    id: str
    dimension: str  # architecture, performance, coverage, etc.
    severity: Severity
    category: str  # complexity, security, coverage, etc.
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    recommendation: str = ""
    effort_hours: float = 0.0
    priority: Priority = Priority.P2
    role: Role = Role.BACKEND
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['severity'] = self.severity.value
        d['priority'] = self.priority.value
        d['role'] = self.role.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Issue':
        """Create from dictionary."""
        data = data.copy()
        data['severity'] = Severity(data['severity'])
        data['priority'] = Priority(data['priority'])
        data['role'] = Role(data['role'])
        return cls(**data)


@dataclass
class ArchitectureAnalysis:
    """Architecture quality analysis results."""
    total_files: int = 0
    large_files: List[Dict[str, Any]] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    coupling_metrics: Dict[str, Any] = field(default_factory=dict)
    solid_violations: List[Issue] = field(default_factory=list)
    score: float = 0.0


@dataclass
class PerformanceAnalysis:
    """Performance profiling results."""
    gpu_utilization: float = 0.0
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    flame_graph_path: str = ""
    memory_usage_peak_gb: float = 0.0
    memory_usage_avg_gb: float = 0.0
    score: float = 0.0


@dataclass
class CoverageAnalysis:
    """Test coverage analysis results."""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    untested_critical_paths: List[str] = field(default_factory=list)
    missing_property_tests: List[str] = field(default_factory=list)
    flaky_tests: List[str] = field(default_factory=list)
    slow_tests: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0


@dataclass
class CodeQualityAnalysis:
    """Code quality metrics."""
    average_complexity: float = 0.0
    high_complexity_functions: List[Dict[str, Any]] = field(default_factory=list)
    duplication_percentage: float = 0.0
    documentation_coverage: float = 0.0
    pylint_score: float = 0.0
    score: float = 0.0
    fix_suggestions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DependencyAnalysis:
    """Dependency security and health."""
    total_dependencies: int = 0
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    outdated_packages: List[str] = field(default_factory=list)
    license_issues: List[str] = field(default_factory=list)
    unused_dependencies: List[str] = field(default_factory=list)
    redundant_dependencies: List[str] = field(default_factory=list)
    security_report: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class DeploymentAnalysis:
    """Deployment readiness assessment."""
    dockerfile_score: float = 0.0
    k8s_readiness: float = 0.0
    ci_cd_completeness: float = 0.0
    monitoring_score: float = 0.0
    score: float = 0.0


@dataclass
class SecurityAnalysis:
    """Security vulnerability assessment."""
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    hipaa_compliance_score: float = 0.0
    hardcoded_secrets: List[str] = field(default_factory=list)
    injection_risks: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0


@dataclass
class ScalabilityAnalysis:
    """Scalability assessment."""
    ddp_correctness: bool = False
    scaling_efficiency: str = "unknown"  # linear, sub-linear, super-linear
    memory_bottlenecks: List[str] = field(default_factory=list)
    communication_overhead_ms: float = 0.0
    score: float = 0.0


@dataclass
class AnalysisResult:
    """Unified analysis result from all dimensions."""
    timestamp: str
    project_path: str
    git_commit: str
    architecture: ArchitectureAnalysis
    performance: PerformanceAnalysis
    coverage: CoverageAnalysis
    code_quality: CodeQualityAnalysis
    dependencies: DependencyAnalysis
    deployment: DeploymentAnalysis
    security: SecurityAnalysis
    scalability: ScalabilityAnalysis
    overall_score: float = 0.0
    critical_issues: List[Issue] = field(default_factory=list)

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """Return JSON schema for validation."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["timestamp", "project_path", "git_commit", "architecture", "performance",
                        "coverage", "code_quality", "dependencies", "deployment", "security",
                        "scalability", "overall_score"],
            "properties": {
                "timestamp": {"type": "string"},
                "project_path": {"type": "string"},
                "git_commit": {"type": "string"},
                "architecture": {"type": "object"},
                "performance": {"type": "object"},
                "coverage": {"type": "object"},
                "code_quality": {"type": "object"},
                "dependencies": {"type": "object"},
                "deployment": {"type": "object"},
                "security": {"type": "object"},
                "scalability": {"type": "object"},
                "overall_score": {"type": "number", "minimum": 0, "maximum": 100},
                "critical_issues": {"type": "array"}
            }
        }

    def to_json(self, validate_schema: bool = True) -> str:
        """Serialize to JSON format with optional schema validation."""
        def convert_value(obj):
            if isinstance(obj, (ArchitectureAnalysis, PerformanceAnalysis, CoverageAnalysis,
                               CodeQualityAnalysis, DependencyAnalysis, DeploymentAnalysis,
                               SecurityAnalysis, ScalabilityAnalysis)):
                return asdict(obj)
            elif isinstance(obj, Issue):
                return obj.to_dict()
            elif isinstance(obj, (Severity, Priority, Role)):
                return obj.value
            return obj

        data = asdict(self)
        # Convert nested dataclasses
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = value
            elif isinstance(value, list):
                data[key] = [convert_value(item) if isinstance(item, Issue) else item for item in value]

        # Validate against schema
        if validate_schema:
            try:
                validate(instance=data, schema=self.get_json_schema())
            except ValidationError as e:
                raise ValueError(f"Schema validation failed: {e.message}") from e

        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str, validate_schema: bool = True) -> 'AnalysisResult':
        """Deserialize from JSON format with optional schema validation."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}") from e

        # Validate against schema
        if validate_schema:
            try:
                validate(instance=data, schema=cls.get_json_schema())
            except ValidationError as e:
                raise ValueError(f"Schema validation failed: {e.message}") from e

        # Convert nested structures
        try:
            data['architecture'] = ArchitectureAnalysis(**data['architecture'])
            data['performance'] = PerformanceAnalysis(**data['performance'])
            data['coverage'] = CoverageAnalysis(**data['coverage'])
            data['code_quality'] = CodeQualityAnalysis(**data['code_quality'])
            data['dependencies'] = DependencyAnalysis(**data['dependencies'])
            data['deployment'] = DeploymentAnalysis(**data['deployment'])
            data['security'] = SecurityAnalysis(**data['security'])
            data['scalability'] = ScalabilityAnalysis(**data['scalability'])

            # Convert issues
            if 'critical_issues' in data:
                data['critical_issues'] = [Issue.from_dict(issue) for issue in data['critical_issues']]

            return cls(**data)
        except (TypeError, KeyError) as e:
            raise ValueError(f"Failed to deserialize AnalysisResult: {str(e)}") from e


@dataclass
class Task:
    """Individual optimization task."""
    id: str
    title: str
    description: str
    priority: Priority
    effort_hours: float
    role: Role
    dependencies: List[str] = field(default_factory=list)
    success_criteria: str = ""
    implementation_guide: str = ""
    references: List[str] = field(default_factory=list)


@dataclass
class OptimizationPlan:
    """Actionable task list with prioritization."""
    tasks: List[Task]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    total_effort_hours: float = 0.0
    estimated_completion_weeks: float = 0.0

    def to_gantt_chart(self) -> str:
        """Generate Gantt chart visualization (placeholder)."""
        # TODO: Implement Gantt chart generation using matplotlib/plotly
        return "Gantt chart generation not yet implemented"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tasks': [asdict(task) for task in self.tasks],
            'dependencies': self.dependencies,
            'total_effort_hours': self.total_effort_hours,
            'estimated_completion_weeks': self.estimated_completion_weeks
        }
