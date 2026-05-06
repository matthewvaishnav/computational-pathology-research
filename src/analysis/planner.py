"""
Optimization Planner for HistoCore Project Optimization Analysis System.

Creates actionable optimization plans with task prioritization, effort estimation,
dependency resolution, and role assignment based on analysis results.
"""

import logging
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta

from .models import (
    AnalysisResult,
    OptimizationPlan,
    Task,
    Issue,
    Priority,
    Severity,
    Role
)

logger = logging.getLogger(__name__)


class OptimizationPlanner:
    """Creates comprehensive optimization plans from analysis results."""
    
    def __init__(self):
        """Initialize optimization planner."""
        self.effort_multipliers = {
            Severity.CRITICAL: 1.5,  # Critical issues take longer
            Severity.HIGH: 1.2,
            Severity.MEDIUM: 1.0,
            Severity.LOW: 0.8
        }
        
        self.role_capacity = {
            Role.BACKEND: 40,    # hours per week
            Role.ML: 40,
            Role.DEVOPS: 30,
            Role.SECURITY: 20,
            Role.QA: 35
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
        
        # Generate tasks from analysis results
        all_tasks = self._generate_tasks_from_analysis(result)
        
        # Assign priorities based on severity and impact
        prioritized_tasks = self._assign_priorities(all_tasks, result)
        
        # Estimate effort for each task
        tasks_with_effort = self._estimate_effort(prioritized_tasks)
        
        # Resolve dependencies between tasks
        dependency_graph = self._resolve_dependencies(tasks_with_effort)
        
        # Assign roles based on task type and workload balancing
        tasks_with_roles = self._assign_roles(tasks_with_effort)
        
        # Generate implementation guides
        final_tasks = self._generate_implementation_guides(tasks_with_roles)
        
        # Calculate totals
        total_effort = sum(task.effort_hours for task in final_tasks)
        estimated_weeks = self._calculate_timeline(final_tasks, dependency_graph)
        
        plan = OptimizationPlan(
            tasks=final_tasks,
            dependencies=dependency_graph,
            total_effort_hours=total_effort,
            estimated_completion_weeks=estimated_weeks
        )
        
        logger.info(f"Created plan with {len(final_tasks)} tasks, "
                   f"{total_effort:.1f} hours, {estimated_weeks:.1f} weeks")
        
        return plan
    
    def _generate_tasks_from_analysis(self, result: AnalysisResult) -> List[Task]:
        """Generate tasks from analysis findings."""
        tasks = []
        task_id = 1
        
        # Architecture tasks
        tasks.extend(self._generate_architecture_tasks(result.architecture, task_id))
        task_id += len(tasks)
        
        # Performance tasks
        tasks.extend(self._generate_performance_tasks(result.performance, task_id))
        task_id += len(tasks)
        
        # Coverage tasks
        tasks.extend(self._generate_coverage_tasks(result.coverage, task_id))
        task_id += len(tasks)
        
        # Code quality tasks
        tasks.extend(self._generate_code_quality_tasks(result.code_quality, task_id))
        task_id += len(tasks)
        
        # Dependencies tasks
        tasks.extend(self._generate_dependency_tasks(result.dependencies, task_id))
        task_id += len(tasks)
        
        # Deployment tasks
        tasks.extend(self._generate_deployment_tasks(result.deployment, task_id))
        task_id += len(tasks)
        
        # Security tasks
        tasks.extend(self._generate_security_tasks(result.security, task_id))
        task_id += len(tasks)
        
        # Scalability tasks
        tasks.extend(self._generate_scalability_tasks(result.scalability, task_id))
        
        return tasks
    
    def _generate_architecture_tasks(self, arch, start_id: int) -> List[Task]:
        """Generate architecture improvement tasks."""
        tasks = []
        
        # Large file refactoring
        if len(arch.large_files) > 0:
            tasks.append(Task(
                id=f"ARCH-{start_id}",
                title="Refactor Large Files",
                description=f"Break down {len(arch.large_files)} large files (>500 lines) into smaller, more maintainable modules",
                priority=Priority.P1,
                effort_hours=len(arch.large_files) * 4.0,  # 4 hours per large file
                role=Role.BACKEND,
                success_criteria="All files under 500 lines, maintain functionality",
                implementation_guide="Use Extract Class and Extract Method refactoring patterns"
            ))
        
        # Circular dependency resolution
        if len(arch.circular_dependencies) > 0:
            tasks.append(Task(
                id=f"ARCH-{start_id + 1}",
                title="Resolve Circular Dependencies",
                description=f"Fix {len(arch.circular_dependencies)} circular dependency cycles",
                priority=Priority.P0,
                effort_hours=len(arch.circular_dependencies) * 6.0,
                role=Role.BACKEND,
                success_criteria="Zero circular dependencies detected",
                implementation_guide="Use Dependency Inversion Principle and interface abstractions"
            ))
        
        # SOLID violations
        if len(arch.solid_violations) > 0:
            high_priority_violations = [v for v in arch.solid_violations if v.severity in [Severity.CRITICAL, Severity.HIGH]]
            if high_priority_violations:
                tasks.append(Task(
                    id=f"ARCH-{start_id + 2}",
                    title="Fix SOLID Principle Violations",
                    description=f"Address {len(high_priority_violations)} high-priority SOLID violations",
                    priority=Priority.P1,
                    effort_hours=len(high_priority_violations) * 3.0,
                    role=Role.BACKEND,
                    success_criteria="Reduce SOLID violations by 80%",
                    implementation_guide="Apply Single Responsibility and Open/Closed principles"
                ))
        
        return tasks
    
    def _generate_performance_tasks(self, perf, start_id: int) -> List[Task]:
        """Generate performance optimization tasks."""
        tasks = []
        
        # GPU utilization improvement
        if perf.gpu_utilization < 70:
            tasks.append(Task(
                id=f"PERF-{start_id}",
                title="Optimize GPU Utilization",
                description=f"Improve GPU utilization from {perf.gpu_utilization:.1f}% to >80%",
                priority=Priority.P1,
                effort_hours=16.0,
                role=Role.ML,
                success_criteria="GPU utilization >80% during training",
                implementation_guide="Optimize batch sizes, data loading, and model parallelization"
            ))
        
        # Memory optimization
        if perf.memory_usage_peak_gb > 12:  # Assuming 16GB GPU
            tasks.append(Task(
                id=f"PERF-{start_id + 1}",
                title="Reduce Memory Usage",
                description=f"Optimize memory usage from {perf.memory_usage_peak_gb:.1f}GB peak",
                priority=Priority.P2,
                effort_hours=12.0,
                role=Role.ML,
                success_criteria="Peak memory usage <12GB",
                implementation_guide="Implement gradient checkpointing and mixed precision training"
            ))
        
        # Bottleneck resolution
        if len(perf.bottlenecks) > 0:
            critical_bottlenecks = [b for b in perf.bottlenecks if b.get('time_ms', 0) > 1000]
            if critical_bottlenecks:
                tasks.append(Task(
                    id=f"PERF-{start_id + 2}",
                    title="Fix Performance Bottlenecks",
                    description=f"Optimize {len(critical_bottlenecks)} critical performance bottlenecks",
                    priority=Priority.P1,
                    effort_hours=len(critical_bottlenecks) * 4.0,
                    role=Role.ML,
                    success_criteria="No functions taking >1000ms",
                    implementation_guide="Profile and optimize hot paths, consider caching and parallelization"
                ))
        
        return tasks
    
    def _generate_coverage_tasks(self, cov, start_id: int) -> List[Task]:
        """Generate test coverage improvement tasks."""
        tasks = []
        
        # Overall coverage improvement
        if cov.line_coverage < 80:
            tasks.append(Task(
                id=f"COV-{start_id}",
                title="Increase Test Coverage",
                description=f"Improve line coverage from {cov.line_coverage:.1f}% to 80%+",
                priority=Priority.P2,
                effort_hours=20.0,
                role=Role.QA,
                success_criteria="Line coverage >80%, branch coverage >70%",
                implementation_guide="Focus on untested critical paths and edge cases"
            ))
        
        # Critical path testing
        if len(cov.untested_critical_paths) > 0:
            tasks.append(Task(
                id=f"COV-{start_id + 1}",
                title="Test Critical Paths",
                description=f"Add tests for {len(cov.untested_critical_paths)} untested critical paths",
                priority=Priority.P1,
                effort_hours=len(cov.untested_critical_paths) * 2.0,
                role=Role.QA,
                success_criteria="All critical paths have test coverage",
                implementation_guide="Write integration tests for error handling and edge cases"
            ))
        
        # Property-based testing
        if len(cov.missing_property_tests) > 0:
            tasks.append(Task(
                id=f"COV-{start_id + 2}",
                title="Add Property-Based Tests",
                description=f"Implement property tests for {len(cov.missing_property_tests)} data transformation functions",
                priority=Priority.P3,
                effort_hours=len(cov.missing_property_tests) * 1.5,
                role=Role.QA,
                success_criteria="Property tests for all data transformations",
                implementation_guide="Use Hypothesis to test invariants and edge cases"
            ))
        
        # Flaky test fixes
        if len(cov.flaky_tests) > 0:
            tasks.append(Task(
                id=f"COV-{start_id + 3}",
                title="Fix Flaky Tests",
                description=f"Stabilize {len(cov.flaky_tests)} flaky tests",
                priority=Priority.P1,
                effort_hours=len(cov.flaky_tests) * 2.0,
                role=Role.QA,
                success_criteria="Zero flaky tests in CI",
                implementation_guide="Fix timing issues, shared state, and external dependencies"
            ))
        
        return tasks
    
    def _generate_code_quality_tasks(self, qual, start_id: int) -> List[Task]:
        """Generate code quality improvement tasks."""
        tasks = []
        
        # Complexity reduction
        if len(qual.high_complexity_functions) > 0:
            tasks.append(Task(
                id=f"QUAL-{start_id}",
                title="Reduce Code Complexity",
                description=f"Refactor {len(qual.high_complexity_functions)} high-complexity functions",
                priority=Priority.P2,
                effort_hours=len(qual.high_complexity_functions) * 2.0,
                role=Role.BACKEND,
                success_criteria="No functions with complexity >10",
                implementation_guide="Extract methods, reduce nesting, simplify conditionals"
            ))
        
        # Documentation improvement
        if qual.documentation_coverage < 80:
            tasks.append(Task(
                id=f"QUAL-{start_id + 1}",
                title="Improve Documentation",
                description=f"Increase documentation coverage from {qual.documentation_coverage:.1f}% to 80%+",
                priority=Priority.P3,
                effort_hours=15.0,
                role=Role.BACKEND,
                success_criteria="80%+ functions have docstrings",
                implementation_guide="Add Google-style docstrings with examples"
            ))
        
        # Code duplication removal
        if qual.duplication_percentage > 5:
            tasks.append(Task(
                id=f"QUAL-{start_id + 2}",
                title="Remove Code Duplication",
                description=f"Reduce code duplication from {qual.duplication_percentage:.1f}% to <5%",
                priority=Priority.P2,
                effort_hours=12.0,
                role=Role.BACKEND,
                success_criteria="Code duplication <5%",
                implementation_guide="Extract common functions and create utility modules"
            ))
        
        return tasks
    
    def _generate_dependency_tasks(self, deps, start_id: int) -> List[Task]:
        """Generate dependency management tasks."""
        tasks = []
        
        # Security vulnerabilities
        if len(deps.vulnerabilities) > 0:
            critical_vulns = [v for v in deps.vulnerabilities if v.get('severity') in ['critical', 'high']]
            if critical_vulns:
                tasks.append(Task(
                    id=f"DEP-{start_id}",
                    title="Fix Security Vulnerabilities",
                    description=f"Address {len(critical_vulns)} critical/high security vulnerabilities",
                    priority=Priority.P0,
                    effort_hours=len(critical_vulns) * 2.0,
                    role=Role.SECURITY,
                    success_criteria="Zero critical/high vulnerabilities",
                    implementation_guide="Update packages, apply patches, or find alternatives"
                ))
        
        # Outdated packages
        if len(deps.outdated_packages) > 10:
            tasks.append(Task(
                id=f"DEP-{start_id + 1}",
                title="Update Dependencies",
                description=f"Update {len(deps.outdated_packages)} outdated packages",
                priority=Priority.P2,
                effort_hours=8.0,
                role=Role.DEVOPS,
                success_criteria="All packages within 2 major versions of latest",
                implementation_guide="Test updates in staging, check for breaking changes"
            ))
        
        return tasks
    
    def _generate_deployment_tasks(self, deploy, start_id: int) -> List[Task]:
        """Generate deployment improvement tasks."""
        tasks = []
        
        # Dockerfile optimization
        if deploy.dockerfile_score < 70:
            tasks.append(Task(
                id=f"DEPLOY-{start_id}",
                title="Optimize Dockerfile",
                description=f"Improve Dockerfile score from {deploy.dockerfile_score:.1f} to 80+",
                priority=Priority.P2,
                effort_hours=6.0,
                role=Role.DEVOPS,
                success_criteria="Dockerfile score >80",
                implementation_guide="Multi-stage builds, layer caching, security best practices"
            ))
        
        # CI/CD pipeline enhancement
        if deploy.ci_cd_completeness < 80:
            tasks.append(Task(
                id=f"DEPLOY-{start_id + 1}",
                title="Enhance CI/CD Pipeline",
                description=f"Improve CI/CD completeness from {deploy.ci_cd_completeness:.1f}% to 80%+",
                priority=Priority.P2,
                effort_hours=12.0,
                role=Role.DEVOPS,
                success_criteria="Complete CI/CD with security scanning and deployment",
                implementation_guide="Add security scans, automated testing, deployment stages"
            ))
        
        # Monitoring setup
        if deploy.monitoring_score < 60:
            tasks.append(Task(
                id=f"DEPLOY-{start_id + 2}",
                title="Implement Monitoring",
                description=f"Set up comprehensive monitoring and observability",
                priority=Priority.P3,
                effort_hours=16.0,
                role=Role.DEVOPS,
                success_criteria="Full application and infrastructure monitoring",
                implementation_guide="Prometheus, Grafana, log aggregation, alerting"
            ))
        
        return tasks
    
    def _generate_security_tasks(self, sec, start_id: int) -> List[Task]:
        """Generate security improvement tasks."""
        tasks = []
        
        # Security vulnerabilities
        if len(sec.vulnerabilities) > 0:
            tasks.append(Task(
                id=f"SEC-{start_id}",
                title="Fix Security Issues",
                description=f"Address {len(sec.vulnerabilities)} security vulnerabilities",
                priority=Priority.P0,
                effort_hours=len(sec.vulnerabilities) * 3.0,
                role=Role.SECURITY,
                success_criteria="Zero security vulnerabilities",
                implementation_guide="Input validation, secure coding practices, penetration testing"
            ))
        
        # HIPAA compliance
        if sec.hipaa_compliance_score < 80:
            tasks.append(Task(
                id=f"SEC-{start_id + 1}",
                title="Improve HIPAA Compliance",
                description=f"Enhance HIPAA compliance from {sec.hipaa_compliance_score:.1f}% to 90%+",
                priority=Priority.P1,
                effort_hours=20.0,
                role=Role.SECURITY,
                success_criteria="HIPAA compliance score >90%",
                implementation_guide="Audit logging, access controls, encryption, data retention"
            ))
        
        # Hardcoded secrets
        if len(sec.hardcoded_secrets) > 0:
            tasks.append(Task(
                id=f"SEC-{start_id + 2}",
                title="Remove Hardcoded Secrets",
                description=f"Replace {len(sec.hardcoded_secrets)} hardcoded secrets with secure storage",
                priority=Priority.P0,
                effort_hours=len(sec.hardcoded_secrets) * 1.0,
                role=Role.SECURITY,
                success_criteria="Zero hardcoded secrets in codebase",
                implementation_guide="Use environment variables, key vaults, or secret management"
            ))
        
        return tasks
    
    def _generate_scalability_tasks(self, scale, start_id: int) -> List[Task]:
        """Generate scalability improvement tasks."""
        tasks = []
        
        # DDP implementation fixes
        if not scale.ddp_correctness:
            tasks.append(Task(
                id=f"SCALE-{start_id}",
                title="Fix DDP Implementation",
                description="Correct DistributedDataParallel implementation issues",
                priority=Priority.P1,
                effort_hours=12.0,
                role=Role.ML,
                success_criteria="DDP implementation passes all correctness checks",
                implementation_guide="Fix gradient synchronization, proper initialization, barrier handling"
            ))
        
        # Communication overhead optimization
        if scale.communication_overhead_ms > 100:
            tasks.append(Task(
                id=f"SCALE-{start_id + 1}",
                title="Reduce Communication Overhead",
                description=f"Optimize communication overhead from {scale.communication_overhead_ms:.1f}ms",
                priority=Priority.P2,
                effort_hours=8.0,
                role=Role.ML,
                success_criteria="Communication overhead <50ms",
                implementation_guide="Gradient compression, efficient all-reduce, overlap computation"
            ))
        
        return tasks
    
    def _assign_priorities(self, tasks: List[Task], result: AnalysisResult) -> List[Task]:
        """Assign priorities based on severity and business impact."""
        # Priority assignment is already done in task generation
        # This method can be used for additional priority refinement
        
        # Boost security and critical architecture tasks
        for task in tasks:
            if task.role == Role.SECURITY and 'vulnerabilities' in task.description.lower():
                task.priority = Priority.P0
            elif 'circular' in task.description.lower():
                task.priority = Priority.P0
            elif 'flaky' in task.description.lower():
                task.priority = Priority.P1
        
        return tasks
    
    def _estimate_effort(self, tasks: List[Task]) -> List[Task]:
        """Refine effort estimates based on complexity and dependencies."""
        # Effort estimation is already done in task generation
        # This method can be used for additional refinement
        
        for task in tasks:
            # Add complexity multiplier based on role
            if task.role == Role.SECURITY:
                task.effort_hours *= 1.3  # Security tasks are more complex
            elif task.role == Role.ML:
                task.effort_hours *= 1.2  # ML tasks need experimentation
            elif task.role == Role.DEVOPS:
                task.effort_hours *= 1.1  # DevOps tasks need testing
        
        return tasks
    
    def _resolve_dependencies(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Build dependency graph between tasks."""
        dependencies = {}
        
        # Define common dependency patterns
        for task in tasks:
            task_deps = []
            
            # Security tasks should come first
            if task.role != Role.SECURITY:
                security_tasks = [t.id for t in tasks if t.role == Role.SECURITY and t.priority == Priority.P0]
                task_deps.extend(security_tasks)
            
            # Architecture tasks should come before performance optimization
            if 'performance' in task.title.lower() or 'optimize' in task.title.lower():
                arch_tasks = [t.id for t in tasks if 'refactor' in t.title.lower() or 'circular' in t.title.lower()]
                task_deps.extend(arch_tasks)
            
            # Testing tasks should come after code changes
            if task.role == Role.QA and 'coverage' in task.title.lower():
                code_tasks = [t.id for t in tasks if t.role in [Role.BACKEND, Role.ML] and t.id != task.id]
                task_deps.extend(code_tasks[:2])  # Depend on first 2 code tasks
            
            # Deployment tasks should come after code quality improvements
            if task.role == Role.DEVOPS:
                quality_tasks = [t.id for t in tasks if 'quality' in t.title.lower() or 'test' in t.title.lower()]
                task_deps.extend(quality_tasks)
            
            dependencies[task.id] = list(set(task_deps))  # Remove duplicates
        
        return dependencies
    
    def _assign_roles(self, tasks: List[Task]) -> List[Task]:
        """Balance workload across roles."""
        # Role assignment is already done in task generation
        # This method can be used for workload balancing
        
        role_workload = defaultdict(float)
        
        # Calculate current workload per role
        for task in tasks:
            role_workload[task.role] += task.effort_hours
        
        # Rebalance if needed (simple heuristic)
        for task in tasks:
            if role_workload[task.role] > 60:  # Overloaded role
                # Try to reassign to less loaded role if compatible
                if task.role == Role.BACKEND and role_workload[Role.ML] < 40:
                    if 'performance' in task.title.lower():
                        task.role = Role.ML
                        role_workload[Role.BACKEND] -= task.effort_hours
                        role_workload[Role.ML] += task.effort_hours
        
        return tasks
    
    def _generate_implementation_guides(self, tasks: List[Task]) -> List[Task]:
        """Add detailed implementation guides and success criteria."""
        # Implementation guides are already added in task generation
        # This method can be used for additional details
        
        for task in tasks:
            # Add references based on task type
            if 'refactor' in task.title.lower():
                task.references = [
                    "https://refactoring.guru/refactoring/techniques",
                    "Clean Code by Robert Martin"
                ]
            elif 'security' in task.title.lower():
                task.references = [
                    "OWASP Top 10",
                    "HIPAA Security Rule"
                ]
            elif 'performance' in task.title.lower():
                task.references = [
                    "PyTorch Performance Tuning Guide",
                    "NVIDIA Deep Learning Performance Guide"
                ]
            elif 'test' in task.title.lower():
                task.references = [
                    "pytest documentation",
                    "Hypothesis property-based testing"
                ]
        
        return tasks
    
    def _calculate_timeline(self, tasks: List[Task], dependencies: Dict[str, List[str]]) -> float:
        """Calculate estimated completion timeline in weeks."""
        # Topological sort to get execution order
        sorted_tasks = self._topological_sort(tasks, dependencies)
        
        # Calculate parallel execution timeline
        role_schedules = defaultdict(float)  # Current end time for each role
        
        for task in sorted_tasks:
            # Find when this task can start (after dependencies complete)
            start_time = 0.0
            for dep_id in dependencies.get(task.id, []):
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task:
                    dep_end_time = role_schedules[dep_task.role]
                    start_time = max(start_time, dep_end_time)
            
            # Schedule task after dependencies and current role workload
            actual_start = max(start_time, role_schedules[task.role])
            task_duration_weeks = task.effort_hours / self.role_capacity.get(task.role, 40)
            role_schedules[task.role] = actual_start + task_duration_weeks
        
        # Total timeline is when the last role finishes
        return max(role_schedules.values()) if role_schedules else 0.0
    
    def _topological_sort(self, tasks: List[Task], dependencies: Dict[str, List[str]]) -> List[Task]:
        """Topologically sort tasks based on dependencies."""
        # Build task lookup
        task_map = {task.id: task for task in tasks}
        
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for task_id, deps in dependencies.items():
            for dep in deps:
                in_degree[task_id] += 1
        
        # Initialize queue with tasks having no dependencies
        queue = deque([task for task in tasks if in_degree[task.id] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree for dependent tasks
            for task_id, deps in dependencies.items():
                if current.id in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_map[task_id])
        
        # If we couldn't sort all tasks, there's a cycle
        if len(result) != len(tasks):
            logger.warning("Circular dependency detected in task graph")
            return tasks  # Return original order
        
        return result
    
    def generate_gantt_data(self, plan: OptimizationPlan) -> Dict[str, Any]:
        """
        Generate data for Gantt chart visualization.
        
        Args:
            plan: Optimization plan
            
        Returns:
            Dictionary with Gantt chart data
        """
        # Calculate task schedules
        role_schedules = defaultdict(float)
        task_schedules = []
        
        # Sort tasks by dependencies
        sorted_tasks = self._topological_sort(plan.tasks, plan.dependencies)
        
        start_date = datetime.now()
        
        for task in sorted_tasks:
            # Calculate start time
            dep_end_time = 0.0
            for dep_id in plan.dependencies.get(task.id, []):
                dep_schedule = next((s for s in task_schedules if s['id'] == dep_id), None)
                if dep_schedule:
                    dep_end_time = max(dep_end_time, dep_schedule['end_week'])
            
            start_week = max(dep_end_time, role_schedules[task.role])
            duration_weeks = task.effort_hours / self.role_capacity.get(task.role, 40)
            end_week = start_week + duration_weeks
            
            task_schedules.append({
                'id': task.id,
                'title': task.title,
                'role': task.role.value,
                'priority': task.priority.value,
                'start_week': start_week,
                'end_week': end_week,
                'duration_weeks': duration_weeks,
                'effort_hours': task.effort_hours,
                'start_date': (start_date + timedelta(weeks=start_week)).isoformat(),
                'end_date': (start_date + timedelta(weeks=end_week)).isoformat()
            })
            
            role_schedules[task.role] = end_week
        
        return {
            'tasks': task_schedules,
            'total_weeks': max(s['end_week'] for s in task_schedules) if task_schedules else 0,
            'roles': list(set(s['role'] for s in task_schedules)),
            'start_date': start_date.isoformat(),
            'end_date': (start_date + timedelta(weeks=plan.estimated_completion_weeks)).isoformat()
        }
