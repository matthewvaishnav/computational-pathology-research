# Implementation Plan: HistoCore Project Optimization Analysis System

## Overview

This implementation plan breaks down the Project Optimization Analysis System into discrete coding tasks. The system analyzes the HistoCore codebase across 8 dimensions (Architecture, Performance, Coverage, Code Quality, Dependencies, Deployment, Security, Scalability) and generates actionable optimization reports with prioritized task lists.

**Implementation Approach**:
1. **Core Infrastructure First**: Data models, orchestrator, serialization
2. **Analyzers in Parallel**: 8 independent analyzer implementations
3. **Reporting Layer**: JSON/Markdown/HTML report generation with visualizations
4. **Optimization Planner**: Task prioritization, effort estimation, dependency resolution
5. **CI/CD Integration**: GitHub Actions workflow for automated analysis
6. **Testing Throughout**: Unit tests, integration tests, snapshot tests

**Technology Stack**: Python 3.11+, pytest, radon, pylint, bandit, safety, pytest-cov, matplotlib, plotly

## Tasks

### Phase 1: Core Infrastructure

- [x] 1. Set up project structure and core data models
  - Create `src/analysis/` directory structure
  - Define `AnalysisResult`, `Issue`, `OptimizationPlan` dataclasses in `src/analysis/models.py`
  - Implement JSON schema validation using `jsonschema`
  - Add type hints for all data models
  - _Requirements: 11.1, 11.2, 11.6_

- [x]* 1.1 Write unit tests for data models
  - Test dataclass initialization and validation
  - Test edge cases (empty results, missing fields)
  - _Requirements: 11.1, 11.2_

- [x] 2. Implement JSON serialization and deserialization
  - Implement `AnalysisResult.to_json()` method with datetime handling
  - Implement `AnalysisResult.from_json()` classmethod with error handling
  - Add schema validation before serialization
  - Handle nested objects (all 8 analysis dimensions)
  - _Requirements: 11.1, 11.2, 11.5, 11.6_

- [x]* 2.1 Write property test for round-trip serialization
  - **Property 1: Round-trip consistency** - FOR ALL valid AnalysisResult objects, parse(serialize(obj)) == obj
  - **Validates: Requirements 11.4**
  - Use Hypothesis to generate random AnalysisResult objects
  - Verify no data loss during serialization/deserialization

- [x]* 2.2 Write unit tests for serialization error handling
  - Test invalid JSON input (malformed, missing fields)
  - Test schema validation failures
  - Verify descriptive error messages
  - _Requirements: 11.5_

- [x] 3. Implement Analysis Orchestrator
  - Create `src/analysis/orchestrator.py` with main `analyze_project()` function
  - Implement parallel analyzer execution using `concurrent.futures.ThreadPoolExecutor`
  - Add error recovery for partial results (graceful degradation)
  - Implement timing and resource usage tracking
  - Add CLI interface with `argparse` (--output, --format, --parallel)
  - _Requirements: 9.1, 9.2_

- [x]* 3.1 Write integration tests for orchestrator
  - Test parallel execution of multiple analyzers
  - Test error recovery when one analyzer fails
  - Test CLI argument parsing
  - _Requirements: 9.1_

- [x] 4. Checkpoint - Verify core infrastructure
  - Ensure all tests pass, ask the user if questions arise.

### Phase 2: Architecture Analyzer

- [ ] 5. Implement Architecture Analyzer
  - [x] 5.1 Create `src/analysis/architecture.py` with `ArchitectureAnalyzer` class
    - Implement `analyze()` method returning `ArchitectureAnalysis` dataclass
    - Integrate `radon cc` for cyclomatic complexity measurement
    - Integrate `radon mi` for maintainability index
    - _Requirements: 1.1, 1.2_

  - [x] 5.2 Implement large file detection
    - Scan all Python files for line counts
    - Flag files >500 lines as refactoring candidates
    - Calculate complexity metrics for large files
    - _Requirements: 1.2_

  - [x] 5.3 Implement circular dependency detection
    - Build import graph using AST parsing
    - Detect cycles using depth-first search
    - Generate cycle paths for reporting
    - _Requirements: 1.3_

  - [x] 5.4 Implement coupling metrics calculation
    - Calculate fan-in (number of modules importing this module)
    - Calculate fan-out (number of modules this module imports)
    - Identify high-coupling modules (fan-out >10)
    - _Requirements: 1.4_

  - [x] 5.5 Implement SOLID principle violation detection
    - Detect Single Responsibility violations (classes >500 lines)
    - Detect Open/Closed violations (excessive conditionals)
    - Generate recommendations with severity ratings
    - _Requirements: 1.5, 1.6, 1.7, 1.8_

- [x]* 5.6 Write unit tests for Architecture Analyzer
  - Test large file detection with sample files
  - Test circular dependency detection with mock import graphs
  - Test coupling metrics calculation
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

### Phase 3: Performance Profiler

- [ ] 6. Implement Performance Profiler
  - [x] 6.1 Create `src/analysis/performance.py` with `PerformanceProfiler` class
    - Implement `analyze()` method returning `PerformanceAnalysis` dataclass
    - Integrate `cProfile` for CPU profiling
    - Integrate `torch.profiler` for GPU profiling
    - _Requirements: 2.1, 2.6_

  - [x] 6.2 Implement GPU utilization measurement
    - Use `torch.cuda` APIs to measure GPU memory usage
    - Use `nvidia-smi` subprocess calls for utilization percentage
    - Track peak and average GPU utilization
    - _Requirements: 2.1_

  - [x] 6.3 Implement data loading bottleneck detection
    - Profile `torch.utils.data.DataLoader` operations
    - Identify CPU-bound operations (>100ms per batch)
    - Measure data loading vs training time ratio
    - _Requirements: 2.2, 2.4_

  - [x] 6.4 Implement flame graph generation
    - Use `py-spy` to generate sampling profiles
    - Export flame graphs in SVG format
    - Integrate with `speedscope` for interactive visualization
    - _Requirements: 2.6_

  - [x] 6.5 Generate optimization recommendations
    - Compare against nnMIL baseline (276 samples/sec)
    - Calculate expected speedup percentages
    - Provide specific optimization strategies
    - _Requirements: 2.5, 2.7, 2.8_

- [x]* 6.6 Write unit tests for Performance Profiler
  - Test GPU utilization measurement with mock data
  - Test bottleneck detection logic
  - Test recommendation generation
  - _Requirements: 2.1, 2.2, 2.4_

### Phase 4: Coverage Analyzer

- [ ] 7. Implement Coverage Analyzer
  - [x] 7.1 Create `src/analysis/coverage.py` with `CoverageAnalyzer` class
    - Implement `analyze()` method returning `CoverageAnalysis` dataclass
    - Integrate `pytest-cov` for line/branch coverage
    - Parse coverage JSON output from `coverage.py`
    - _Requirements: 3.1_

  - [x] 7.2 Implement untested critical path detection
    - Use AST parsing to identify error handling blocks
    - Detect uncovered exception handlers
    - Identify edge cases without tests
    - _Requirements: 3.2_

  - [x] 7.3 Implement missing property test detection
    - Scan for data transformation functions
    - Identify functions suitable for property-based testing
    - Check if property tests exist in test suite
    - _Requirements: 3.3_

  - [x] 7.4 Implement test quality metrics
    - Calculate assertion density (assertions per test)
    - Detect test isolation issues (shared state)
    - Identify flaky tests from historical CI data
    - Measure test execution time
    - _Requirements: 3.4, 3.5, 3.6_

  - [x] 7.5 Generate prioritized coverage gap list
    - Assign risk ratings (critical, high, medium, low)
    - Suggest specific test cases for each gap
    - Estimate effort for closing gaps
    - _Requirements: 3.7, 3.8_

- [x]* 7.6 Write unit tests for Coverage Analyzer
  - Test coverage data parsing
  - Test critical path detection with sample AST
  - Test quality metrics calculation
  - _Requirements: 3.1, 3.2, 3.4_

### Phase 5: Code Quality Scanner

- [ ] 8. Implement Code Quality Scanner
  - [x] 8.1 Create `src/analysis/code_quality.py` with `CodeQualityScanner` class
    - Implement `analyze()` method returning `CodeQualityAnalysis` dataclass
    - Integrate `pylint` for code quality checks
    - Integrate `flake8` for style violations
    - Integrate `mypy` for type checking
    - _Requirements: 4.1, 4.5_

  - [x] 8.2 Implement complexity analysis
    - Use `radon cc` to compute cyclomatic complexity
    - Flag functions with complexity >10
    - Calculate average complexity per module
    - _Requirements: 4.1, 4.2_

  - [x] 8.3 Implement code duplication detection
    - Use AST comparison for structural similarity
    - Detect duplicates (>10 lines, >80% similarity)
    - Generate refactoring recommendations
    - _Requirements: 4.3_

  - [x] 8.4 Implement documentation coverage measurement
    - Scan for missing docstrings
    - Check type hint coverage
    - Verify docstring format (Google/NumPy style)
    - _Requirements: 4.4_

  - [x] 8.5 Implement automated fix suggestions
    - Generate fixes for unused imports
    - Suggest naming convention corrections
    - Provide refactoring templates
    - Calculate module quality scores (0-100)
    - _Requirements: 4.6, 4.7, 4.8_

- [x]* 8.6 Write unit tests for Code Quality Scanner
  - Test complexity calculation
  - Test duplication detection with sample code
  - Test documentation coverage measurement
  - _Requirements: 4.1, 4.3, 4.4_

### Phase 6: Dependency Auditor

- [ ] 9. Implement Dependency Auditor
  - [x] 9.1 Create `src/analysis/dependencies.py` with `DependencyAuditor` class
    - Implement `analyze()` method returning `DependencyAnalysis` dataclass
    - Parse `requirements.txt` and `pyproject.toml`
    - Integrate `pip-audit` for CVE scanning
    - Integrate `safety` for vulnerability detection
    - _Requirements: 5.1, 5.2_

  - [x] 9.2 Implement outdated package detection
    - Use `pip list --outdated` to find updates
    - Check for security patches in new versions
    - Prioritize security updates over feature updates
    - _Requirements: 5.3_

  - [x] 9.3 Implement dependency bloat detection
    - Analyze import usage across codebase
    - Identify unused dependencies
    - Detect redundant packages (overlapping functionality)
    - _Requirements: 5.4_

  - [x] 9.4 Implement license compatibility validation
    - Use `pip-licenses` to extract license info
    - Check compatibility (MIT, Apache 2.0, BSD)
    - Flag GPL/AGPL licenses for review
    - _Requirements: 5.5_

  - [x] 9.5 Generate security report with upgrade paths
    - Include CVSS scores for vulnerabilities
    - Provide upgrade commands
    - Suggest workarounds for unpatchable issues
    - _Requirements: 5.6, 5.7, 5.8_

- [x]* 9.6 Write unit tests for Dependency Auditor
  - Test requirements.txt parsing
  - Test CVE detection with mock data
  - Test license compatibility validation
  - _Requirements: 5.1, 5.2, 5.5_

### Phase 7: Deployment Validator

- [x] 10. Implement Deployment Validator
  - [x] 10.1 Create `src/analysis/deployment.py` with `DeploymentValidator` class
    - Implement `analyze()` method returning `DeploymentAnalysis` dataclass
    - Integrate `hadolint` for Dockerfile linting
    - Parse Kubernetes manifests from `k8s/` directory
    - _Requirements: 6.1, 6.2_

  - [x] 10.2 Implement Dockerfile best practices validation
    - Check for multi-stage builds
    - Verify layer caching optimization
    - Detect security issues (running as root)
    - _Requirements: 6.1_

  - [x] 10.3 Implement Kubernetes manifest validation
    - Use `kubeval` for schema validation
    - Check resource limits (CPU, memory)
    - Verify health checks (liveness, readiness probes)
    - _Requirements: 6.2_

  - [x] 10.4 Implement CI/CD pipeline assessment
    - Parse `.github/workflows/` YAML files
    - Verify build, test, deploy stages exist
    - Check for security scanning steps
    - _Requirements: 6.3_

  - [x] 10.5 Generate deployment readiness score
    - Calculate score (0-100) based on best practices
    - Provide implementation guides for gaps
    - Include monitoring and logging recommendations
    - _Requirements: 6.4, 6.5, 6.6, 6.7, 6.8_

- [x]* 10.6 Write unit tests for Deployment Validator
  - Test Dockerfile parsing and validation
  - Test Kubernetes manifest validation
  - Test CI/CD pipeline assessment
  - _Requirements: 6.1, 6.2, 6.3_

### Phase 8: Security Scanner

- [x] 11. Implement Security Scanner
  - [x] 11.1 Create `src/analysis/security.py` with `SecurityScanner` class
    - Implement `analyze()` method returning `SecurityAnalysis` dataclass
    - Integrate `bandit` for Python security scanning
    - Parse bandit JSON output
    - _Requirements: 7.1, 7.2_

  - [x] 11.2 Implement injection vulnerability detection
    - Detect SQL injection patterns
    - Detect command injection in subprocess calls
    - Flag unsafe `eval()` and `exec()` usage
    - _Requirements: 7.1, 7.2_

  - [x] 11.3 Implement TLS/SSL configuration validation
    - Check PACS integration for TLS usage
    - Verify federated learning encryption
    - Detect insecure SSL contexts
    - _Requirements: 7.3_

  - [x] 11.4 Implement hardcoded secrets detection
    - Scan for API keys, passwords, tokens
    - Use regex patterns for common secret formats
    - Check for private keys in codebase
    - _Requirements: 7.4_

  - [x] 11.5 Implement HIPAA compliance checklist
    - Verify audit logging (7-year retention)
    - Check input sanitization
    - Validate access controls
    - Generate gap analysis report
    - _Requirements: 7.5, 7.6, 7.7, 7.8_

- [x]* 11.6 Write unit tests for Security Scanner
  - Test injection detection with sample code
  - Test secrets detection with mock patterns
  - Test HIPAA compliance checklist generation
  - _Requirements: 7.1, 7.2, 7.4_

### Phase 9: Scalability Analyzer

- [x] 12. Implement Scalability Analyzer
  - [x] 12.1 Create `src/analysis/scalability.py` with `ScalabilityAnalyzer` class
    - Implement `analyze()` method returning `ScalabilityAnalysis` dataclass
    - Verify DistributedDataParallel (DDP) implementation
    - Check for proper `torch.distributed` usage
    - _Requirements: 8.1_

  - [x] 12.2 Implement data loading bottleneck detection
    - Analyze DataLoader configuration for multi-GPU
    - Check for proper worker count and prefetching
    - Identify serialization bottlenecks
    - _Requirements: 8.2_

  - [x] 12.3 Implement communication overhead measurement
    - Estimate gradient synchronization time
    - Check for gradient accumulation usage
    - Identify all-reduce bottlenecks
    - _Requirements: 8.3, 8.4_

  - [x] 12.4 Implement large dataset handling assessment
    - Check for streaming data loading (>1TB datasets)
    - Verify memory-efficient WSI processing
    - Identify memory bottlenecks for gigapixel images
    - _Requirements: 8.5, 8.6_

  - [x] 12.5 Generate scaling recommendations
    - Classify scaling efficiency (linear, sub-linear, super-linear)
    - Provide optimization strategies for multi-GPU
    - Estimate speedup for 2/4/8 GPU configurations
    - _Requirements: 8.7, 8.8_

- [x]* 12.6 Write unit tests for Scalability Analyzer
  - Test DDP implementation verification
  - Test communication overhead estimation
  - Test scaling recommendation generation
  - _Requirements: 8.1, 8.3, 8.7_

- [x] 13. Checkpoint - Verify all analyzers complete
  - Ensure all tests pass, ask the user if questions arise.

### Phase 10: Report Generation

- [x] 14. Implement Result Aggregator
  - Create `src/analysis/aggregator.py` with `ResultAggregator` class
  - Merge results from all 8 analyzers into unified `AnalysisResult`
  - Resolve conflicts and deduplicate findings
  - Compute overall health score (0-100)
  - Extract top 10 critical issues across all dimensions
  - _Requirements: 9.1, 9.2_

- [x]* 14.1 Write unit tests for Result Aggregator
  - Test merging of multiple analyzer results
  - Test deduplication logic
  - Test overall score calculation
  - _Requirements: 9.1_

- [x] 15. Implement Markdown Report Generator
  - [x] 15.1 Create `src/analysis/reporting.py` with `ReportGenerator` class
    - Implement `generate_markdown()` method
    - Format executive summary with top 10 issues
    - Generate detailed sections for all 8 dimensions
    - _Requirements: 9.1, 9.2, 9.7_

  - [x] 15.2 Implement quantitative metrics formatting
    - Format coverage percentages, complexity scores
    - Generate comparison tables (before/after nnMIL)
    - Include vulnerability counts and severity distributions
    - _Requirements: 9.3, 9.6_

  - [x] 15.3 Implement qualitative assessment formatting
    - Format architecture quality descriptions
    - Include code maintainability narratives
    - Add contextual explanations for metrics
    - _Requirements: 9.4_

  - [x] 15.4 Generate prioritized task list
    - Format tasks by priority (P0, P1, P2, P3)
    - Include effort estimates and dependencies
    - Add implementation examples for each task
    - _Requirements: 9.5, 9.8_

- [x]* 15.5 Write unit tests for Markdown Report Generator
  - Test Markdown formatting with sample data
  - Test table generation
  - Test task list formatting
  - _Requirements: 9.7_

- [x] 16. Implement visualization generation
  - [x] 16.1 Create `src/analysis/visualizations.py` with chart generation functions
    - Use `matplotlib` for static charts (bar, pie, line)
    - Use `plotly` for interactive charts
    - Generate coverage heatmaps
    - Generate complexity distribution histograms
    - _Requirements: 9.3_

  - [x] 16.2 Implement HTML/PDF export
    - Use `pandoc` to convert Markdown to HTML
    - Use `weasyprint` to convert HTML to PDF
    - Embed visualizations in HTML reports
    - Add CSS styling for professional appearance
    - _Requirements: 9.7_

- [x]* 16.3 Write integration tests for report generation
  - Test end-to-end report generation (JSON → Markdown → HTML)
  - Test visualization embedding
  - Test PDF export (if pandoc/weasyprint available)
  - _Requirements: 9.7_

### Phase 11: Optimization Planner

- [x] 17. Implement Optimization Planner
  - [x] 17.1 Create `src/analysis/planner.py` with `OptimizationPlanner` class
    - Implement `create_plan()` method returning `OptimizationPlan`
    - Categorize tasks by priority (P0, P1, P2, P3)
    - Assign severity-based priorities (critical→P0, high→P1, etc.)
    - _Requirements: 10.1_

  - [x] 17.2 Implement effort estimation
    - Use complexity metrics for estimation
    - Apply historical effort data (if available)
    - Estimate hours/days/weeks for each task
    - Calculate total effort and timeline
    - _Requirements: 10.2_

  - [x] 17.3 Implement dependency resolution
    - Build task dependency graph
    - Identify sequencing constraints
    - Detect circular dependencies in tasks
    - Generate topologically sorted task order
    - _Requirements: 10.3_

  - [x] 17.4 Implement role assignment
    - Assign tasks to roles (backend, ML, DevOps, security)
    - Balance workload across roles
    - Identify tasks requiring multiple roles
    - _Requirements: 10.4_

  - [x] 17.5 Generate implementation guides
    - Include code examples for each task
    - Add links to relevant documentation
    - Provide success criteria and validation methods
    - _Requirements: 10.5, 10.6_

  - [x] 17.6 Implement Gantt chart generation
    - Use `matplotlib` or `plotly` for timeline visualization
    - Show task dependencies and critical path
    - Export as PNG/SVG/HTML
    - _Requirements: 10.7_

- [x]* 17.7 Write unit tests for Optimization Planner
  - Test priority assignment logic
  - Test effort estimation calculation
  - Test dependency resolution
  - Test role assignment
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

### Phase 12: CI/CD Integration

- [x] 18. Implement Regression Detector
  - [x] 18.1 Create `src/analysis/regression_detector.py` with `RegressionDetector` class
    - Implement `detect_regressions()` comparing baseline vs current
    - Load baseline analysis results from JSON
    - Compare coverage percentages (flag >2% decrease)
    - _Requirements: 12.2_

  - [x] 18.2 Implement performance regression detection
    - Compare performance metrics (flag >10% slowdown)
    - Identify new bottlenecks not in baseline
    - Generate performance diff report
    - _Requirements: 12.3_

  - [x] 18.3 Implement security regression detection
    - Compare vulnerability counts
    - Flag new CVEs introduced in PR
    - Check for new hardcoded secrets
    - _Requirements: 12.4_

  - [x] 18.4 Generate diff reports
    - Format side-by-side comparison (baseline vs current)
    - Highlight regressions in red, improvements in green
    - Include root cause analysis for regressions
    - _Requirements: 12.5, 12.8_

  - [x] 18.5 Implement CI build failure logic
    - Return non-zero exit code for critical regressions
    - Allow configurable thresholds (coverage, performance)
    - Generate actionable error messages
    - _Requirements: 12.6_

- [x]* 18.6 Write unit tests for Regression Detector
  - Test coverage regression detection
  - Test performance regression detection
  - Test security regression detection
  - Test CI failure logic
  - _Requirements: 12.2, 12.3, 12.4, 12.6_

- [-] 19. Create GitHub Actions workflow
  - [x] 19.1 Create `.github/workflows/project-analysis.yml`
    - Add workflow triggers (push, pull_request)
    - Install analysis dependencies
    - Run analysis orchestrator
    - Upload analysis results as artifacts
    - _Requirements: 12.1_

  - [x] 19.2 Implement PR comment posting
    - Use `actions/github-script` to post comments
    - Format analysis summary for PR comments
    - Include links to full reports
    - Only post on pull_request events
    - _Requirements: 12.7_

  - [x] 19.3 Implement baseline comparison
    - Fetch baseline results from main branch
    - Run regression detector
    - Post diff report as PR comment
    - Fail workflow if critical regressions detected
    - _Requirements: 12.5, 12.6_

- [x]* 19.4 Write integration tests for CI workflow
  - Test workflow execution in test repository
  - Test PR comment posting (mock GitHub API)
  - Test regression detection in CI context
  - _Requirements: 12.1, 12.7_

- [x] 20. Checkpoint - Verify CI/CD integration
  - Ensure all tests pass, ask the user if questions arise.

### Phase 13: Testing and Documentation

- [x] 21. Implement snapshot tests for report generation
  - Create `tests/analysis/test_snapshots.py`
  - Capture expected report output for known codebase
  - Use `pytest-snapshot` or manual JSON comparison
  - Detect regressions in report format
  - _Requirements: 9.7_

- [x]* 21.1 Write integration tests for end-to-end analysis
  - Create small test project (10-20 files)
  - Run full analysis pipeline
  - Verify all 8 analyzers execute successfully
  - Check report generation in all formats
  - _Requirements: 9.1, 9.7_

- [x] 22. Create comprehensive documentation
  - [x] 22.1 Write `src/analysis/README.md` with usage guide
    - Document CLI usage and options
    - Provide example commands
    - Explain output formats
    - _Requirements: 9.8_

  - [x] 22.2 Write `docs/ANALYSIS_SYSTEM.md` with architecture overview
    - Document system components and data flow
    - Explain analyzer implementations
    - Provide troubleshooting guide
    - _Requirements: 9.8_

  - [x] 22.3 Add docstrings to all public functions
    - Use Google-style docstrings
    - Include parameter types and return types
    - Add usage examples in docstrings
    - _Requirements: 9.8_

- [x] 23. Create requirements file for analysis tools
  - Create `requirements-analysis.txt` with all tool dependencies
  - Pin versions for reproducibility
  - Add installation instructions to README
  - Test installation in clean virtual environment
  - _Requirements: 9.1_

- [x] 24. Final checkpoint - Complete system validation
  - Run full analysis on HistoCore codebase
  - Verify all 8 dimensions produce results
  - Generate reports in all formats (JSON, Markdown, HTML, PDF)
  - Validate round-trip serialization
  - Test CI/CD integration with sample PR
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- **No Optional Tasks**: All sub-tasks are required for MVP (no `*` postfix) because this is an infrastructure/tooling system without property-based testing requirements
- **Parallel Implementation**: Analyzers (Phase 2-9) can be implemented in parallel after Phase 1 completes
- **Tool Dependencies**: Requires external tools (radon, pylint, bandit, safety, pytest-cov, hadolint) - installation tested in Phase 13
- **Incremental Validation**: Checkpoints after each major phase ensure system stability
- **Requirements Traceability**: Each task explicitly references requirements for validation
- **Effort Estimates**: Based on design complexity and tool integration overhead
  - Phase 1 (Core Infrastructure): ~16 hours
  - Phase 2-9 (8 Analyzers): ~80 hours (10h each)
  - Phase 10 (Reporting): ~24 hours
  - Phase 11 (Optimization Planner): ~16 hours
  - Phase 12 (CI/CD Integration): ~16 hours
  - Phase 13 (Testing & Docs): ~16 hours
  - **Total Estimated Effort**: ~168 hours (~4 weeks for 1 developer)

## Success Criteria

- All 8 analyzers execute successfully on HistoCore codebase
- Reports generated in JSON, Markdown, HTML, PDF formats
- Round-trip serialization preserves all data (no loss)
- CI/CD integration detects regressions automatically
- Optimization plan includes prioritized tasks with effort estimates
- All unit tests and integration tests pass
- Documentation complete with usage examples
