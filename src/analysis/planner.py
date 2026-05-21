"""
Optimization Planner for HistoCore Project Optimization Analysis System.

Creates actionable optimization plans with task prioritization, effort estimation,
dependency resolution, and role assignment based on analysis results.
"""

import logging
from collections import defaultdict
from typing import Dict, List

from .models import AnalysisResult, OptimizationPlan, Role, Severity, Task

logger = logging.getLogger(__name__)


class OptimizationPlanner:
    """Creates comprehensive optimization plans from analysis results."""

    def __init__(self):
        """Initialize optimization planner."""
        self.effort_multipliers = {
            Severity.CRITICAL: 1.5,  # Critical issues take longer
            Severity.HIGH: 1.2,
            Severity.MEDIUM: 1.0,
            Severity.LOW: 0.8,
        }

        self.role_capacity = {
            Role.BACKEND: 40,  # hours per week
            Role.ML: 40,
            Role.DEVOPS: 30,
            Role.SECURITY: 20,
            Role.QA: 35,
        }

    def create_plan(self, result: AnalysisResult) -> OptimizationPlan:
        """
        Create comprehensive optimization plan from analysis results.

        Args:
            result: Analysis result containing all findings

        Returns:
            OptimizationPlan with prioritized tasks and dependencies
        """
        logger.info("Creating optimization plan...")

        # Convert issues to tasks
        tasks = self._convert_issues_to_tasks(result.critical_issues)

        # Resolve dependencies between tasks
        dependency_graph = self._resolve_dependencies(tasks)

        # Calculate totals
        total_effort = sum(task.effort_hours for task in tasks)
        estimated_weeks = self._calculate_timeline(tasks, dependency_graph)

        plan = OptimizationPlan(
            tasks=tasks,
            dependencies=dependency_graph,
            total_effort_hours=total_effort,
            estimated_completion_weeks=estimated_weeks,
        )

        logger.info(
            f"Created plan with {len(tasks)} tasks, "
            f"{total_effort:.1f} hours, {estimated_weeks:.1f} weeks"
        )

        return plan

    def _convert_issues_to_tasks(self, issues: List) -> List[Task]:
        """Convert issues to tasks with implementation guides."""
        tasks = []

        for i, issue in enumerate(issues, 1):
            # Generate implementation guide based on issue type
            impl_guide = self._generate_implementation_guide(issue)

            # Generate success criteria
            success_criteria = self._generate_success_criteria(issue)

            task = Task(
                id=f"task-{i}",
                title=issue.title,
                description=issue.description,
                priority=issue.priority,
                effort_hours=issue.effort_hours,
                role=issue.role,
                dependencies=[],  # Will be resolved later
                success_criteria=success_criteria,
                implementation_guide=impl_guide,
                references=issue.references,
            )
            tasks.append(task)

        return tasks

    def _generate_implementation_guide(self, issue) -> str:
        """Generate implementation guide based on issue category."""
        guides = {
            "cve": f"""1. Review CVE details and affected versions
2. Update dependency: pip install --upgrade {issue.file_path.replace('requirements.txt', '')}
3. Run tests to verify compatibility
4. Update requirements.txt with new version""",
            "vulnerability": f"""1. Review security vulnerability in {issue.file_path}
2. {issue.recommendation}
3. Add security tests to prevent regression
4. Run security scanner to verify fix""",
            "untested_critical_path": f"""1. Identify test scenarios for {issue.file_path}
2. Write unit tests covering error handling
3. Add edge case tests
4. Verify coverage increase with pytest-cov""",
            "complexity": f"""1. Analyze function structure in {issue.file_path}
2. Extract helper functions for complex logic
3. Refactor using design patterns
4. Verify complexity reduction with radon""",
            "bottleneck": f"""1. Profile code section to identify hot spots
2. {issue.recommendation}
3. Benchmark before/after performance
4. Document optimization in code comments""",
        }

        return guides.get(
            issue.category,
            f"""1. Review issue in {issue.file_path}
2. {issue.recommendation}
3. Test changes thoroughly
4. Document solution""",
        )

    def _generate_success_criteria(self, issue) -> str:
        """Generate success criteria for task."""
        criteria = {
            "cve": "CVE resolved, all tests pass, no new vulnerabilities introduced",
            "vulnerability": "Security scan passes, vulnerability no longer detected",
            "untested_critical_path": "Coverage increased by >5%, all new tests pass",
            "complexity": "Cyclomatic complexity reduced below 10",
            "bottleneck": "Performance improved by >10%, benchmarks confirm speedup",
        }

        return criteria.get(issue.category, "Issue resolved, tests pass, no regressions")

    def _resolve_dependencies(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """
        Identify task dependencies based on file paths and categories.

        Args:
            tasks: List of tasks

        Returns:
            Dictionary mapping task IDs to list of dependency task IDs
        """
        dependencies = defaultdict(list)

        # Simple dependency rules:
        # 1. Security tasks should be done before other tasks in same file
        # 2. Architecture refactoring before performance optimization
        # 3. Test coverage before code quality improvements

        for i, task in enumerate(tasks):
            for j, other_task in enumerate(tasks):
                if i == j:
                    continue

                # Security dependencies
                if (
                    task.role == Role.SECURITY
                    and other_task.role != Role.SECURITY
                    and task.priority.value < other_task.priority.value
                ):
                    dependencies[other_task.id].append(task.id)

                # Architecture before performance
                if (
                    "architecture" in task.title.lower()
                    and "performance" in other_task.title.lower()
                ):
                    dependencies[other_task.id].append(task.id)

                # Coverage before quality
                if "coverage" in task.title.lower() and "quality" in other_task.title.lower():
                    dependencies[other_task.id].append(task.id)

        return dict(dependencies)

    def _calculate_timeline(self, tasks: List[Task], dependencies: Dict[str, List[str]]) -> float:
        """
        Calculate estimated completion timeline in weeks.

        Args:
            tasks: List of tasks
            dependencies: Task dependency graph

        Returns:
            Estimated weeks to completion
        """
        # Group tasks by role
        role_tasks = defaultdict(list)
        for task in tasks:
            role_tasks[task.role].append(task)

        # Calculate parallel execution time per role
        max_weeks = 0.0
        for role, role_task_list in role_tasks.items():
            role_hours = sum(t.effort_hours for t in role_task_list)
            capacity = self.role_capacity.get(role, 40)
            role_weeks = role_hours / capacity
            max_weeks = max(max_weeks, role_weeks)

        # Add 20% buffer for dependencies and coordination
        return max_weeks * 1.2

    def _topological_sort(
        self, tasks: List[Task], dependencies: Dict[str, List[str]]
    ) -> List[Task]:
        """
        Sort tasks in topological order respecting dependencies.

        Args:
            tasks: List of tasks
            dependencies: Task dependency graph

        Returns:
            Topologically sorted task list
        """
        # Build adjacency list
        task_map = {task.id: task for task in tasks}
        in_degree = {task.id: 0 for task in tasks}

        for task_id, deps in dependencies.items():
            in_degree[task_id] = len(deps)

        # Find tasks with no dependencies
        queue = [task for task in tasks if in_degree[task.id] == 0]
        sorted_tasks = []

        while queue:
            # Sort by priority
            queue.sort(key=lambda t: (t.priority.value, -t.effort_hours))
            current = queue.pop(0)
            sorted_tasks.append(current)

            # Update in-degrees
            for task_id, deps in dependencies.items():
                if current.id in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_map[task_id])

        # Check for cycles
        if len(sorted_tasks) != len(tasks):
            logger.warning("Circular dependencies detected, using priority sort")
            return sorted(tasks, key=lambda t: (t.priority.value, -t.effort_hours))

        return sorted_tasks
