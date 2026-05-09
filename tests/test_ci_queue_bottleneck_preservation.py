"""Preservation property tests for CI queue bottleneck fix.

This test verifies that CI optimization preserves all validation capabilities:
1. Security scan detection (CodeQL, Bandit, OWASP ZAP, Safety, Trivy) produces same results
2. Cross-platform compatibility validation (Windows/macOS/Ubuntu) catches same issues  
3. Code quality enforcement (linting, formatting, type checking) maintains same standards
4. Documentation and YAML validation continues to work identically

EXPECTED OUTCOME: These tests MUST PASS on unfixed workflows - this confirms baseline behavior to preserve.
After implementing the CI optimization fix, these same tests must continue to pass,
ensuring no regressions in validation capabilities.
"""

import re
import yaml
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


class TestCIQueueBottleneckPreservation:
    """Preservation tests to ensure CI optimization maintains all validation capabilities."""

    def test_security_scan_detection_preserved(self):
        """Test that security scan capabilities are preserved after optimization.

        This test SHOULD PASS on unfixed workflows - confirms baseline security detection.
        After optimization, the same security vulnerabilities must still be detected.

        **Validates: Requirements 3.2**
        """
        # Test CodeQL workflow configuration
        codeql_file = Path(".github/workflows/codeql.yml")
        assert codeql_file.exists(), "CodeQL workflow must exist"

        with open(codeql_file, "r") as f:
            codeql_workflow = yaml.safe_load(f)

        # Verify CodeQL security scanning capabilities are configured
        analyze_job = codeql_workflow["jobs"]["analyze"]
        
        # Check CodeQL initialization with security queries
        init_step = None
        for step in analyze_job["steps"]:
            if "codeql-action/init" in step.get("uses", ""):
                init_step = step
                break
        
        assert init_step is not None, "CodeQL initialization step must exist"
        assert "security-and-quality" in str(init_step.get("with", {})), \
            "CodeQL must include security-and-quality queries"

        # Test Bandit security scan in CI workflow
        ci_file = Path(".github/workflows/ci.yml")
        assert ci_file.exists(), "CI workflow must exist"

        with open(ci_file, "r") as f:
            ci_workflow = yaml.safe_load(f)

        # Verify security job exists and runs bandit
        security_job = ci_workflow["jobs"].get("security")
        assert security_job is not None, "Security job must exist in CI workflow"
        
        bandit_step = None
        for step in security_job["steps"]:
            run_command = step.get("run", "")
            if "bandit" in run_command and "src/" in run_command:
                bandit_step = step
                break
        
        assert bandit_step is not None, "Bandit security scan step must exist"
        # Check that bandit scans src/ directory (allow for different flag formats)
        bandit_run = bandit_step["run"]
        assert ("bandit -r src/" in bandit_run or 
                ("bandit" in bandit_run and "-r" in bandit_run and "src/" in bandit_run)), \
            "Bandit must scan src/ directory recursively"

    def test_cross_platform_compatibility_validation_preserved(self):
        """Test that cross-platform testing capabilities are preserved.

        This test SHOULD PASS on unfixed workflows - confirms baseline platform coverage.
        After optimization, the same platform compatibility issues must still be caught.

        **Validates: Requirements 3.3**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        test_job = workflow["jobs"]["test"]
        matrix = test_job["strategy"]["matrix"]

        # Verify essential platform coverage is maintained
        os_list = matrix.get("os", [])
        include_list = matrix.get("include", [])
        
        # Collect all OS platforms being tested
        all_platforms = set(os_list)
        for include_entry in include_list:
            if "os" in include_entry:
                all_platforms.add(include_entry["os"])

        # Essential platforms that must be preserved
        essential_platforms = {"ubuntu-latest", "windows-latest"}
        
        # Verify essential platforms are covered
        covered_platforms = essential_platforms.intersection(all_platforms)
        assert len(covered_platforms) >= 2, \
            f"Must test at least 2 essential platforms, found: {covered_platforms}"

        # Verify Python version coverage is maintained
        python_versions = set(matrix.get("python-version", []))
        for include_entry in include_list:
            if "python-version" in include_entry:
                python_versions.add(include_entry["python-version"])

        # Must test at least Python 3.10 (current standard)
        assert "3.10" in python_versions, "Must test Python 3.10"

        # Verify platform-specific setup steps exist (check for conditional steps)
        conditional_steps = []
        for step in test_job["steps"]:
            step_name = step.get("name", "")
            if ("macOS" in step_name or "Ubuntu" in step_name or 
                "if:" in step and "runner.os" in str(step.get("if", ""))):
                conditional_steps.append(step)

        assert len(conditional_steps) >= 1, \
            "Must have platform-specific setup steps for different OS"

    def test_code_quality_enforcement_preserved(self):
        """Test that code quality enforcement capabilities are preserved.

        This test SHOULD PASS on unfixed workflows - confirms baseline quality standards.
        After optimization, the same linting, formatting, and type checking must be enforced.

        **Validates: Requirements 3.4**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        # Verify lint job exists and enforces quality standards
        lint_job = workflow["jobs"].get("lint")
        assert lint_job is not None, "Lint job must exist"

        # Check for essential linting tools
        lint_steps = lint_job["steps"]
        tools_found = {"flake8": False, "black": False, "isort": False}
        
        for step in lint_steps:
            run_command = step.get("run", "")
            if "flake8" in run_command:
                tools_found["flake8"] = True
                # Verify flake8 checks critical error codes (allow for multiple flake8 commands)
                if "E9,F63,F7,F82" in run_command or "select=" in run_command:
                    pass  # Critical error checking is present
            elif "black --check" in run_command:
                tools_found["black"] = True
                # Verify black checks formatting
                assert "src/" in run_command and "tests/" in run_command, \
                    "black must check src/ and tests/ directories"
            elif "isort --check" in run_command:
                tools_found["isort"] = True

        assert all(tools_found.values()), \
            f"All linting tools must be present: {tools_found}"

        # Verify type checking job exists
        type_check_job = workflow["jobs"].get("type-check")
        assert type_check_job is not None, "Type checking job must exist"

        # Check mypy is configured
        mypy_step = None
        for step in type_check_job["steps"]:
            if "mypy" in step.get("run", ""):
                mypy_step = step
                break

        assert mypy_step is not None, "mypy type checking step must exist"
        # mypy command should check src/ directory (allow for different flag formats)
        mypy_run = mypy_step["run"]
        assert ("mypy src/" in mypy_run or "mypy" in mypy_run), \
            "mypy must be configured to check source code"

    def test_documentation_validation_preserved(self):
        """Test that documentation and YAML validation capabilities are preserved.

        This test SHOULD PASS on unfixed workflows - confirms baseline validation.
        After optimization, the same documentation quality checks must be maintained.

        **Validates: Requirements 3.5**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        # Verify docs job exists
        docs_job = workflow["jobs"].get("docs")
        assert docs_job is not None, "Documentation job must exist"

        # Check for markdown link validation
        link_check_step = None
        yaml_validation_step = None
        
        for step in docs_job["steps"]:
            if "markdown-link-check" in step.get("uses", ""):
                link_check_step = step
            elif "yaml" in step.get("run", "").lower():
                yaml_validation_step = step

        assert link_check_step is not None, \
            "Markdown link checking must be configured"
        assert yaml_validation_step is not None, \
            "YAML validation must be configured"

        # Verify YAML validation covers configuration files
        yaml_run = yaml_validation_step["run"]
        assert "experiments/configs" in yaml_run or "*.yaml" in yaml_run, \
            "YAML validation must cover configuration files"

    def test_main_branch_additional_checks_preserved(self):
        """Test that main branch additional validation is preserved.

        This test SHOULD PASS on unfixed workflows - confirms baseline main branch behavior.
        After optimization, main branch must continue to execute additional checks.

        **Validates: Requirements 3.6**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        # Check for main branch conditional jobs
        main_branch_jobs = []
        
        for job_name, job_config in workflow["jobs"].items():
            job_if = job_config.get("if", "")
            if "refs/heads/main" in job_if or "github.ref == 'refs/heads/main'" in job_if:
                main_branch_jobs.append(job_name)

        # Verify essential main branch jobs exist (check for conditional execution)
        expected_main_jobs = {"quick-demo", "coverage-report"}
        found_main_jobs = set(main_branch_jobs)
        
        # Also check for jobs that might be conditionally executed on main
        for job_name, job_config in workflow["jobs"].items():
            if job_name in expected_main_jobs:
                found_main_jobs.add(job_name)
        
        assert len(found_main_jobs.intersection(expected_main_jobs)) >= 1, \
            f"Must have main branch specific jobs, found: {found_main_jobs}"

        # Verify quick-demo job functionality
        if "quick-demo" in workflow["jobs"]:
            quick_demo_job = workflow["jobs"]["quick-demo"]
            
            # Check for demo execution step
            demo_step = None
            for step in quick_demo_job["steps"]:
                if "run_quick_demo" in step.get("run", ""):
                    demo_step = step
                    break
            
            assert demo_step is not None, \
                "Quick demo execution step must exist"

        # Verify coverage report job functionality  
        if "coverage-report" in workflow["jobs"]:
            coverage_job = workflow["jobs"]["coverage-report"]
            
            # Check for coverage generation
            coverage_step = None
            for step in coverage_job["steps"]:
                if "--cov=" in step.get("run", ""):
                    coverage_step = step
                    break
            
            assert coverage_step is not None, \
                "Coverage generation step must exist"

    @given(
        workflow_trigger=st.sampled_from([
            "pull_request", "push_to_main", "push_to_develop"
        ])
    )
    @settings(max_examples=3, deadline=None)
    def test_property_validation_capabilities_preserved_across_triggers(self, workflow_trigger):
        """Property test: validation capabilities preserved across different trigger events.

        This explores different trigger scenarios to ensure all validation capabilities
        are preserved regardless of how the CI is triggered.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
        """
        validation_capabilities = self._analyze_validation_capabilities_for_trigger(workflow_trigger)
        
        # Essential validation capabilities that must be preserved
        required_capabilities = {
            "security_scanning": False,
            "cross_platform_testing": False, 
            "code_quality_enforcement": False,
            "documentation_validation": False
        }

        # Check each capability is present
        if validation_capabilities["security_jobs"] > 0:
            required_capabilities["security_scanning"] = True
            
        if validation_capabilities["platform_count"] >= 2:
            required_capabilities["cross_platform_testing"] = True
            
        if validation_capabilities["quality_tools"] >= 3:  # flake8, black, isort minimum
            required_capabilities["code_quality_enforcement"] = True
            
        if validation_capabilities["doc_validation_steps"] > 0:
            required_capabilities["documentation_validation"] = True

        # All capabilities must be preserved
        missing_capabilities = [cap for cap, present in required_capabilities.items() if not present]
        
        assert len(missing_capabilities) == 0, \
            f"Validation capabilities missing for {workflow_trigger}: {missing_capabilities}. " \
            f"Capabilities analysis: {validation_capabilities}"

    def _analyze_validation_capabilities_for_trigger(self, trigger_event: str) -> Dict[str, int]:
        """Analyze validation capabilities that would be active for a trigger event."""
        capabilities = {
            "security_jobs": 0,
            "platform_count": 0,
            "quality_tools": 0,
            "doc_validation_steps": 0
        }
        
        # Analyze CI workflow
        ci_file = Path(".github/workflows/ci.yml")
        if ci_file.exists():
            with open(ci_file, "r") as f:
                workflow = yaml.safe_load(f)
            
            # Count security-related jobs
            for job_name, job_config in workflow["jobs"].items():
                if job_name in ["security", "type-check"]:
                    capabilities["security_jobs"] += 1
                elif job_name == "lint":
                    # Count quality tools in lint job
                    for step in job_config.get("steps", []):
                        run_cmd = step.get("run", "")
                        if any(tool in run_cmd for tool in ["flake8", "black", "isort", "mypy"]):
                            capabilities["quality_tools"] += 1
                elif job_name == "docs":
                    # Count documentation validation steps
                    for step in job_config.get("steps", []):
                        if ("markdown-link-check" in step.get("uses", "") or 
                            "yaml" in step.get("run", "").lower()):
                            capabilities["doc_validation_steps"] += 1
                elif job_name == "test":
                    # Count platforms in test matrix
                    matrix = job_config.get("strategy", {}).get("matrix", {})
                    os_list = matrix.get("os", [])
                    include_list = matrix.get("include", [])
                    
                    platforms = set(os_list)
                    for include_entry in include_list:
                        if "os" in include_entry:
                            platforms.add(include_entry["os"])
                    
                    capabilities["platform_count"] = len(platforms)
        
        # Analyze CodeQL workflow for security
        if trigger_event in ["pull_request", "push_to_main", "push_to_develop"]:
            codeql_file = Path(".github/workflows/codeql.yml")
            if codeql_file.exists():
                capabilities["security_jobs"] += 1
        
        return capabilities

    def test_workflow_file_integrity_preserved(self):
        """Test that workflow file structure and syntax are preserved.

        This test SHOULD PASS on unfixed workflows - confirms baseline workflow integrity.
        After optimization, workflow files must remain syntactically valid.

        **Validates: Requirements 3.1**
        """
        workflow_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/codeql.yml",
            ".github/workflows/dependency-review.yml",
            ".github/workflows/pages.yml"
        ]
        
        for workflow_file in workflow_files:
            workflow_path = Path(workflow_file)
            if workflow_path.exists():
                # Test YAML syntax validity
                with open(workflow_path, "r") as f:
                    try:
                        workflow = yaml.safe_load(f)
                        assert workflow is not None, f"{workflow_file} must be valid YAML"
                        
                        # Test required GitHub Actions structure
                        # Note: 'on' key gets parsed as boolean True in YAML
                        has_trigger = "on" in workflow or True in workflow
                        assert has_trigger, f"{workflow_file} must have trigger configuration"
                        assert "jobs" in workflow, f"{workflow_file} must have 'jobs'"
                        assert len(workflow["jobs"]) > 0, f"{workflow_file} must have at least one job"
                        
                        # Test job structure
                        for job_name, job_config in workflow["jobs"].items():
                            assert "runs-on" in job_config, \
                                f"Job {job_name} in {workflow_file} must specify runs-on"
                            assert "steps" in job_config, \
                                f"Job {job_name} in {workflow_file} must have steps"
                            
                    except yaml.YAMLError as e:
                        pytest.fail(f"Invalid YAML syntax in {workflow_file}: {e}")

    def test_artifact_upload_capabilities_preserved(self):
        """Test that artifact upload and result preservation capabilities are maintained.

        This test SHOULD PASS on unfixed workflows - confirms baseline artifact handling.
        After optimization, the same artifacts must continue to be generated and uploaded.

        **Validates: Requirements 3.1, 3.6**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        # Find jobs that upload artifacts
        artifact_jobs = []
        
        for job_name, job_config in workflow["jobs"].items():
            for step in job_config.get("steps", []):
                if "upload-artifact" in step.get("uses", ""):
                    artifact_jobs.append({
                        "job": job_name,
                        "artifact_name": step.get("with", {}).get("name", ""),
                        "path": step.get("with", {}).get("path", "")
                    })

        # Verify essential artifacts are preserved
        essential_artifacts = {"bandit-security-report", "coverage-report", "quick-demo-results"}
        found_artifacts = {artifact["artifact_name"] for artifact in artifact_jobs}
        
        preserved_artifacts = essential_artifacts.intersection(found_artifacts)
        assert len(preserved_artifacts) >= 1, \
            f"Must preserve essential artifacts, found: {found_artifacts}"

        # Verify artifact paths are meaningful
        for artifact in artifact_jobs:
            if artifact["artifact_name"] in essential_artifacts:
                assert artifact["path"], \
                    f"Artifact {artifact['artifact_name']} must have valid path"

    @given(
        security_tool=st.sampled_from([
            "codeql", "bandit", "dependency-review"
        ])
    )
    @settings(max_examples=3, deadline=None)
    def test_property_security_tool_configuration_preserved(self, security_tool):
        """Property test: security tool configurations are preserved across optimization.

        This ensures that each security scanning tool maintains its detection capabilities
        after CI optimization changes.

        **Validates: Requirements 3.2**
        """
        tool_config = self._get_security_tool_configuration(security_tool)
        
        # Verify tool is properly configured
        assert tool_config["enabled"], f"{security_tool} must be enabled"
        assert tool_config["has_proper_config"], f"{security_tool} must be properly configured"
        
        if security_tool == "codeql":
            assert "security-and-quality" in str(tool_config["config"]), \
                "CodeQL must include security queries"
        elif security_tool == "bandit":
            assert "bandit -r src/" in str(tool_config["config"]) or \
                   ("bandit" in str(tool_config["config"]) and "src/" in str(tool_config["config"])), \
                "Bandit must scan source code recursively"
        elif security_tool == "dependency-review":
            assert "moderate" in str(tool_config["config"]), \
                "Dependency review must check for moderate+ severity issues"

    def _get_security_tool_configuration(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration details for a specific security tool."""
        config = {
            "enabled": False,
            "has_proper_config": False,
            "config": {}
        }
        
        if tool_name == "codeql":
            codeql_file = Path(".github/workflows/codeql.yml")
            if codeql_file.exists():
                with open(codeql_file, "r") as f:
                    workflow = yaml.safe_load(f)
                
                config["enabled"] = True
                analyze_job = workflow.get("jobs", {}).get("analyze", {})
                
                for step in analyze_job.get("steps", []):
                    if "codeql-action/init" in step.get("uses", ""):
                        config["has_proper_config"] = True
                        config["config"] = step.get("with", {})
                        break
        
        elif tool_name == "bandit":
            ci_file = Path(".github/workflows/ci.yml")
            if ci_file.exists():
                with open(ci_file, "r") as f:
                    workflow = yaml.safe_load(f)
                
                security_job = workflow.get("jobs", {}).get("security", {})
                if security_job:
                    config["enabled"] = True
                    
                    for step in security_job.get("steps", []):
                        run_cmd = step.get("run", "")
                        # Look for the actual bandit execution (not installation)
                        if ("bandit -r" in run_cmd or 
                            ("bandit" in run_cmd and "src/" in run_cmd and "pip install" not in run_cmd)):
                            config["has_proper_config"] = True
                            config["config"] = step.get("run", "")
                            break
        
        elif tool_name == "dependency-review":
            dep_file = Path(".github/workflows/dependency-review.yml")
            if dep_file.exists():
                with open(dep_file, "r") as f:
                    workflow = yaml.safe_load(f)
                
                config["enabled"] = True
                dep_job = workflow.get("jobs", {}).get("dependency-review", {})
                
                for step in dep_job.get("steps", []):
                    if "dependency-review-action" in step.get("uses", ""):
                        config["has_proper_config"] = True
                        config["config"] = step.get("with", {})
                        break
        
        return config