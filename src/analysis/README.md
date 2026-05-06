# HistoCore Project Optimization Analysis System

A comprehensive analysis system that evaluates the HistoCore computational pathology project across 8 dimensions of software quality, generating actionable optimization reports with prioritized task lists.

## Overview

The Project Optimization Analysis System provides:

- **Multi-dimensional Analysis**: Evaluates architecture, performance, coverage, code quality, dependencies, deployment, security, and scalability
- **Automated Reporting**: Generates comprehensive reports in Markdown, HTML, and PDF formats
- **Optimization Planning**: Creates prioritized task lists with effort estimates and dependency resolution
- **CI/CD Integration**: Automated analysis with regression detection and PR comments
- **Interactive Visualizations**: Charts, heatmaps, and dashboards for visual insights

## Quick Start

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt
pip install -r requirements-analysis.txt

# Install optional visualization tools
pip install matplotlib plotly weasyprint
```

### Basic Usage

```bash
# Run analysis on current directory
python -m src.analysis.orchestrator

# Run with custom output
python -m src.analysis.orchestrator --output my-analysis.json --format json

# Run with specific options
python -m src.analysis.orchestrator \
  --output results/ \
  --format json \
  --parallel \
  --max-workers 4
```

### Programmatic Usage

```python
from src.analysis.orchestrator import AnalysisOrchestrator
from src.analysis.reporting import ReportGenerator
from src.analysis.planner import OptimizationPlanner

# Run analysis
orchestrator = AnalysisOrchestrator("./")
result = orchestrator.analyze_project()

# Generate reports
generator = ReportGenerator()
markdown_report = generator.generate_markdown(result)
generator.generate_html(result, "report.html")

# Create optimization plan
planner = OptimizationPlanner()
plan = planner.create_plan(result)
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `project_path` | Path to project root directory | `.` (current directory) |
| `--output`, `-o` | Output file path | `analysis.json` |
| `--format`, `-f` | Output format (json, markdown, html, pdf) | `json` |
| `--parallel` | Run analyzers in parallel | `True` |
| `--no-parallel` | Run analyzers sequentially | - |
| `--max-workers` | Maximum parallel workers | `min(8, cpu_count)` |

## Analysis Dimensions

### 1. Architecture Analysis
- **Large file detection** (>500 lines)
- **Circular dependency detection**
- **Coupling metrics** (fan-in/fan-out)
- **SOLID principle violations**
- **Complexity metrics** (cyclomatic complexity, maintainability index)

### 2. Performance Analysis
- **GPU utilization measurement**
- **Memory usage tracking** (peak and average)
- **Bottleneck detection** (CPU profiling)
- **Data loading optimization**
- **Flame graph generation**

### 3. Test Coverage Analysis
- **Line and branch coverage**
- **Untested critical path detection**
- **Missing property test identification**
- **Test quality metrics** (assertion density, isolation)
- **Flaky test detection**

### 4. Code Quality Analysis
- **Complexity analysis** (cyclomatic complexity)
- **Code duplication detection**
- **Documentation coverage**
- **Style violations** (pylint, flake8)
- **Type checking** (mypy)

### 5. Dependency Analysis
- **Security vulnerability scanning** (CVE detection)
- **Outdated package detection**
- **License compatibility validation**
- **Dependency bloat detection**
- **Upgrade path recommendations**

### 6. Deployment Analysis
- **Dockerfile best practices**
- **Kubernetes manifest validation**
- **CI/CD pipeline assessment**
- **Monitoring and observability**
- **Deployment readiness scoring**

### 7. Security Analysis
- **Vulnerability scanning** (bandit)
- **Injection risk detection**
- **Hardcoded secrets detection**
- **HIPAA compliance assessment**
- **TLS/SSL configuration validation**

### 8. Scalability Analysis
- **DistributedDataParallel (DDP) verification**
- **Multi-GPU scaling efficiency**
- **Communication overhead measurement**
- **Large dataset handling assessment**
- **Memory bottleneck identification**

## Output Formats

### JSON Output
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "project_path": "/path/to/project",
  "git_commit": "abc123...",
  "overall_score": 75.2,
  "architecture": {
    "score": 68.5,
    "total_files": 1250,
    "large_files": [...],
    "circular_dependencies": [...]
  },
  "critical_issues": [...]
}
```

### Markdown Report
- Executive summary with key findings
- Dimension-by-dimension analysis
- Critical issues with recommendations
- Prioritized task list
- Visual charts and metrics

### HTML Report
- Interactive dashboard
- Embedded visualizations
- Responsive design
- Exportable to PDF

## CI/CD Integration

### GitHub Actions Workflow

The system includes a complete GitHub Actions workflow (`.github/workflows/project-analysis.yml`) that:

1. **Runs on every PR and push to main**
2. **Generates comprehensive analysis reports**
3. **Compares against baseline for regression detection**
4. **Posts results as PR comments**
5. **Fails builds on critical regressions**
6. **Uploads artifacts for historical tracking**

### Regression Detection

```python
from src.analysis.regression_detector import RegressionDetector

detector = RegressionDetector()
regressions = detector.detect_regressions(current_result, "baseline.json")

if regressions['should_fail_ci']:
    print("Critical regressions detected!")
    exit(1)
```

### Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Line coverage decrease | >2% | Fail build |
| Performance slowdown | >10% | Fail build |
| New security vulnerabilities | Any | Fail build |
| Overall score decrease | >5 points | Fail build |
| Memory increase | >15% | Fail build |

## Optimization Planning

The system generates actionable optimization plans with:

- **Task prioritization** (P0, P1, P2, P3)
- **Effort estimation** (hours/days/weeks)
- **Dependency resolution** (topological sorting)
- **Role assignment** (backend, ML, DevOps, security, QA)
- **Implementation guides** with code examples
- **Gantt chart generation** for timeline visualization

```python
from src.analysis.planner import OptimizationPlanner

planner = OptimizationPlanner()
plan = planner.create_plan(analysis_result)

print(f"Total effort: {plan.total_effort_hours} hours")
print(f"Estimated completion: {plan.estimated_completion_weeks} weeks")
print(f"Tasks: {len(plan.tasks)}")
```

## Visualization

### Static Charts (matplotlib)
- Radar charts for dimension scores
- Bar charts for comparisons
- Heatmaps for coverage analysis
- Histograms for complexity distribution

### Interactive Charts (plotly)
- Interactive dashboards
- Issue priority treemaps
- Performance timelines
- Drill-down capabilities

### Fallback Charts
- ASCII charts for environments without graphics libraries
- Text-based summaries
- CSV exports for data analysis

## Configuration

### Analysis Tools

The system integrates with various analysis tools:

| Tool | Purpose | Installation |
|------|---------|--------------|
| `radon` | Complexity metrics | `pip install radon` |
| `pylint` | Code quality | `pip install pylint` |
| `bandit` | Security scanning | `pip install bandit` |
| `safety` | Vulnerability detection | `pip install safety` |
| `pytest-cov` | Coverage analysis | `pip install pytest-cov` |
| `matplotlib` | Static charts | `pip install matplotlib` |
| `plotly` | Interactive charts | `pip install plotly` |

### Graceful Degradation

The system continues to work even when optional tools are missing:
- **Missing radon**: Uses basic file analysis
- **Missing pylint**: Skips style checking
- **Missing bandit**: Uses basic security checks
- **Missing visualization libraries**: Generates text reports

## Troubleshooting

### Common Issues

1. **Analysis fails with import errors**
   ```bash
   pip install -r requirements-analysis.txt
   ```

2. **Visualization generation fails**
   ```bash
   pip install matplotlib plotly
   # For PDF generation:
   sudo apt-get install pandoc wkhtmltopdf
   ```

3. **Memory issues with large projects**
   ```bash
   # Reduce parallel workers
   python -m src.analysis.orchestrator --max-workers 2
   ```

4. **Permission errors in CI**
   ```yaml
   permissions:
     contents: read
     pull-requests: write
   ```

### Performance Optimization

- **Use parallel execution** (default) for faster analysis
- **Limit max workers** on resource-constrained systems
- **Skip visualization generation** for faster CI runs
- **Use baseline comparison** only on PRs

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check analysis logs:
```bash
python -m src.analysis.orchestrator 2>&1 | tee analysis.log
```

## Examples

### Basic Analysis
```bash
# Analyze current project
python -m src.analysis.orchestrator

# Output: analysis.json with overall score and findings
```

### Generate Reports
```bash
# Generate all report formats
python -c "
from src.analysis.models import AnalysisResult
from src.analysis.reporting import ReportGenerator
from pathlib import Path

result = AnalysisResult.from_json(Path('analysis.json').read_text())
generator = ReportGenerator()

# Generate reports
generator.generate_html(result, 'report.html')
generator.generate_pdf(result, 'report.pdf')
"
```

### CI Integration
```yaml
# .github/workflows/analysis.yml
- name: Run Analysis
  run: python -m src.analysis.orchestrator --output analysis.json
  
- name: Check Regressions
  run: |
    python -c "
    from src.analysis.regression_detector import RegressionDetector
    from src.analysis.models import AnalysisResult
    
    current = AnalysisResult.from_json(open('analysis.json').read())
    detector = RegressionDetector()
    regressions = detector.detect_regressions(current, 'baseline.json')
    
    if regressions['should_fail_ci']:
        exit(1)
    "
```

## API Reference

### Core Classes

- **`AnalysisOrchestrator`**: Main orchestrator for running all analyzers
- **`AnalysisResult`**: Data model for analysis results with JSON serialization
- **`ReportGenerator`**: Generates reports in multiple formats
- **`OptimizationPlanner`**: Creates actionable optimization plans
- **`RegressionDetector`**: Detects regressions between analysis runs
- **`VisualizationGenerator`**: Creates charts and visualizations

### Data Models

- **`Issue`**: Individual finding with severity, priority, and recommendations
- **`Task`**: Optimization task with effort estimates and dependencies
- **`OptimizationPlan`**: Complete plan with tasks, timeline, and Gantt data

See individual module documentation for detailed API reference.

## Contributing

1. **Add new analyzers** by implementing the analyzer interface
2. **Extend report formats** by modifying `ReportGenerator`
3. **Add visualization types** in `VisualizationGenerator`
4. **Improve regression detection** in `RegressionDetector`

## License

This project is part of the HistoCore computational pathology research system.