"""Bug condition exploration test for CI queue bottleneck fix.

This test verifies that the bug condition exists on UNFIXED workflows:
1. CI workflows create 21+ parallel jobs simultaneously when triggered
2. Job matrix expansion creates excessive parallelization without resource limits
3. No job prioritization or concurrency controls exist
4. Multiple workflows trigger independently without coordination

EXPECTED OUTCOME: This test MUST FAIL on unfixed code - failure confirms the bug exists.
The test demonstrates that CI trigger events result in queue times >30 minutes due to
excessive job creation and lack of resource management.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


class TestCIQueueBottleneckBugExploration:
    """Exploratory tests to surface counterexamples demonstrating excessive CI queueing."""

    def test_ci_workflow_creates_excessive_parallel_jobs(self):
        """Test that CI workflow matrix creates 21+ parallel jobs simultaneously.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.
        The current ci.yml creates 5 matrix jobs + 6 additional jobs = 11 jobs,
        plus other workflows add more jobs, totaling 21+ jobs.

        **Validates: Requirements 1.1**
        """
        ci_file = Path(".github/workflows/ci.yml")
        assert ci_file.exists(), f"CI workflow file {ci_file} not found"

        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        # Count jobs in ci.yml
        jobs = workflow.get("jobs", {})
        total_jobs = 0

        for job_name, job_config in jobs.items():
            strategy = job_config.get("strategy", {})
            matrix = strategy.get("matrix", {})
            
            if matrix:
                # Calculate matrix expansion
                matrix_size = 1
                for key, values in matrix.items():
                    if key != "include" and key != "exclude":
                        if isinstance(values, list):
                            matrix_size *= len(values)
                
                # Add include entries
                include_entries = matrix.get("include", [])
                if include_entries:
                    matrix_size += len(include_entries)
                
                total_jobs += matrix_size
            else:
                total_jobs += 1

        # Count jobs from other workflow files
        workflow_dir = Path(".github/workflows")
        other_workflows = [
            "codeql.yml", "dependency-review.yml", "pages.yml", 
            "docker-publish.yml", "release.yml"
        ]
        
        other_jobs = 0
        for workflow_file in other_workflows:
            workflow_path = workflow_dir / workflow_file
            if workflow_path.exists():
                with open(workflow_path, "r") as f:
                    try:
                        other_workflow = yaml.safe_load(f)
                        other_jobs += len(other_workflow.get("jobs", {}))
                    except yaml.YAMLError:
                        continue

        total_concurrent_jobs = total_jobs + other_jobs

        # This assertion SHOULD FAIL on unfixed code
        # The bug condition is when total jobs > reasonable runner capacity (typically 20)
        assert total_concurrent_jobs <= 20, (
            f"Bug confirmed: CI workflows create {total_concurrent_jobs} concurrent jobs "
            f"(ci.yml: {total_jobs}, other workflows: {other_jobs}). "
            f"This exceeds typical GitHub Actions runner capacity and causes queue bottlenecks. "
            f"Jobs will remain queued for 30+ minutes or hours."
        )

    def test_ci_matrix_lacks_resource_optimization(self):
        """Test that CI matrix strategy lacks resource-aware optimization.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.
        The matrix should be optimized to reduce parallel job count.

        **Validates: Requirements 1.2**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        test_job = workflow["jobs"]["test"]
        matrix = test_job["strategy"]["matrix"]

        # Count OS combinations
        os_list = matrix.get("os", [])
        python_list = matrix.get("python-version", [])
        include_list = matrix.get("include", [])

        # Calculate total matrix combinations
        base_combinations = len(os_list) * len(python_list) if os_list and python_list else 0
        total_combinations = base_combinations + len(include_list)

        # Check for resource optimization indicators
        has_fail_fast = test_job.get("strategy", {}).get("fail-fast", True)
        has_concurrency_limits = "concurrency" in workflow
        has_conditional_execution = any(
            "if:" in str(job_config) for job_config in workflow["jobs"].values()
        )

        optimization_score = 0
        if total_combinations <= 3:  # Optimized matrix size
            optimization_score += 1
        if not has_fail_fast:  # fail-fast: false is better for CI reliability
            optimization_score += 1
        if has_concurrency_limits:
            optimization_score += 1
        if has_conditional_execution:
            optimization_score += 1

        # This assertion SHOULD FAIL on unfixed code
        assert optimization_score >= 3, (
            f"Bug confirmed: CI matrix lacks resource optimization. "
            f"Matrix creates {total_combinations} combinations, "
            f"optimization score: {optimization_score}/4. "
            f"Missing optimizations cause excessive resource usage and queue delays."
        )

    def test_workflows_lack_job_prioritization(self):
        """Test that workflows lack job dependencies for prioritization.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.
        Critical jobs should run first, but no 'needs:' dependencies exist.

        **Validates: Requirements 1.3**
        """
        ci_file = Path(".github/workflows/ci.yml")
        with open(ci_file, "r") as f:
            workflow = yaml.safe_load(f)

        jobs = workflow["jobs"]
        
        # Identify critical jobs that should run first
        critical_jobs = ["lint", "type-check", "security"]
        regular_jobs = ["test", "docker"]
        
        # Check for job dependencies using 'needs:' keyword
        prioritization_found = False
        
        for job_name, job_config in jobs.items():
            if job_name in regular_jobs and "needs" in job_config:
                needs = job_config["needs"]
                if isinstance(needs, str):
                    needs = [needs]
                
                # Check if regular jobs depend on critical jobs
                if any(critical_job in needs for critical_job in critical_jobs):
                    prioritization_found = True
                    break

        # This assertion SHOULD FAIL on unfixed code
        assert prioritization_found, (
            f"Bug confirmed: No job prioritization found in CI workflow. "
            f"Critical jobs {critical_jobs} should run before regular jobs {regular_jobs} "
            f"using 'needs:' dependencies, but no such prioritization exists. "
            f"This causes all jobs to queue simultaneously, creating bottlenecks."
        )

    def test_workflows_lack_concurrency_controls(self):
        """Test that workflows lack concurrency groups to prevent resource conflicts.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.
        Multiple workflow runs should be coordinated with concurrency controls.

        **Validates: Requirements 1.4**
        """
        workflow_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/codeql.yml", 
            ".github/workflows/dependency-review.yml"
        ]
        
        workflows_with_concurrency = 0
        total_workflows = 0
        
        for workflow_file in workflow_files:
            workflow_path = Path(workflow_file)
            if workflow_path.exists():
                total_workflows += 1
                with open(workflow_path, "r") as f:
                    try:
                        workflow = yaml.safe_load(f)
                        if "concurrency" in workflow:
                            workflows_with_concurrency += 1
                    except yaml.YAMLError:
                        continue

        concurrency_coverage = workflows_with_concurrency / total_workflows if total_workflows > 0 else 0

        # This assertion SHOULD FAIL on unfixed code
        assert concurrency_coverage >= 0.8, (
            f"Bug confirmed: Only {workflows_with_concurrency}/{total_workflows} workflows "
            f"have concurrency controls ({concurrency_coverage:.1%} coverage). "
            f"Missing concurrency groups allow multiple workflow runs to compete for resources, "
            f"causing queue saturation and delays exceeding 30 minutes."
        )

    @given(
        trigger_event=st.sampled_from([
            "pull_request", "push_to_main", "push_to_develop", "scheduled_codeql"
        ])
    )
    @settings(max_examples=4, deadline=None)
    def test_property_trigger_events_create_excessive_job_queues(self, trigger_event):
        """Property test: verify that CI trigger events create excessive job queues.

        This explores different trigger scenarios to demonstrate queue bottlenecks.
        On unfixed workflows, all trigger types SHOULD create 15+ concurrent jobs.

        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        # Simulate job counting for different trigger events
        job_counts = self._count_jobs_for_trigger_event(trigger_event)
        
        total_jobs = sum(job_counts.values())
        
        # The bug condition: excessive job creation leads to queue times >30 minutes
        # GitHub Actions typically provides 20 concurrent runners for free accounts
        runner_capacity = 20
        
        # This assertion SHOULD FAIL on unfixed code for most trigger events
        assert total_jobs <= runner_capacity, (
            f"Bug confirmed: Trigger event '{trigger_event}' creates {total_jobs} concurrent jobs "
            f"(breakdown: {job_counts}), exceeding typical runner capacity of {runner_capacity}. "
            f"This causes jobs to queue for 30+ minutes or hours, severely impacting development workflow."
        )

    def _count_jobs_for_trigger_event(self, trigger_event: str) -> Dict[str, int]:
        """Count jobs that would be triggered for a specific event."""
        job_counts = {}
        
        # Count ci.yml jobs (always triggered for PR/push events)
        if trigger_event in ["pull_request", "push_to_main", "push_to_develop"]:
            ci_file = Path(".github/workflows/ci.yml")
            if ci_file.exists():
                with open(ci_file, "r") as f:
                    workflow = yaml.safe_load(f)
                
                ci_jobs = 0
                for job_name, job_config in workflow["jobs"].items():
                    strategy = job_config.get("strategy", {})
                    matrix = strategy.get("matrix", {})
                    
                    if matrix:
                        # Calculate matrix expansion for test job
                        if job_name == "test":
                            os_count = len(matrix.get("os", []))
                            python_count = len(matrix.get("python-version", []))
                            include_count = len(matrix.get("include", []))
                            ci_jobs += (os_count * python_count) + include_count
                        else:
                            ci_jobs += 1
                    else:
                        ci_jobs += 1
                
                job_counts["ci.yml"] = ci_jobs
        
        # Count codeql.yml jobs
        if trigger_event in ["pull_request", "push_to_main", "push_to_develop", "scheduled_codeql"]:
            job_counts["codeql.yml"] = 1
        
        # Count dependency-review.yml jobs
        if trigger_event == "pull_request":
            job_counts["dependency-review.yml"] = 1
        
        # Count pages.yml jobs (only on main branch pushes with docs changes)
        if trigger_event == "push_to_main":
            job_counts["pages.yml"] = 2  # build + deploy jobs
        
        return job_counts

    def test_queue_time_simulation_exceeds_30_minutes(self):
        """Test that simulated queue times exceed 30 minutes with current job load.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.
        Simulates GitHub Actions queue behavior with current job configuration.

        **Validates: Requirements 1.1, 1.2**
        """
        # Simulate a typical PR scenario
        total_jobs = self._count_jobs_for_trigger_event("pull_request")
        concurrent_jobs = sum(total_jobs.values())
        
        # GitHub Actions runner assumptions (conservative estimates)
        available_runners = 5  # Typical for free accounts during peak hours
        job_duration_minutes = 15  # Average job duration
        
        # Simple queue simulation
        if concurrent_jobs <= available_runners:
            max_queue_time = 0
        else:
            # Jobs beyond runner capacity must wait
            queued_jobs = concurrent_jobs - available_runners
            # Assuming jobs complete in waves
            waves_needed = (queued_jobs + available_runners - 1) // available_runners
            max_queue_time = waves_needed * job_duration_minutes
        
        # This assertion SHOULD FAIL on unfixed code
        assert max_queue_time <= 30, (
            f"Bug confirmed: Simulated queue time is {max_queue_time} minutes "
            f"with {concurrent_jobs} concurrent jobs and {available_runners} available runners. "
            f"This exceeds the 30-minute threshold, confirming that jobs experience "
            f"excessive queue delays in the current CI configuration."
        )