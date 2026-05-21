"""
Clinical Impact Measurement Module for Medical AI Revolution
Tracks diagnostic accuracy, turnaround times, user satisfaction, and clinical outcomes.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class DiagnosticAccuracyMetric:
    """Diagnostic accuracy measurement."""

    case_id: str
    ai_prediction: str
    pathologist_diagnosis: str
    ground_truth: str
    confidence_score: float
    processing_time_seconds: float
    timestamp: datetime
    site_id: str


@dataclass
class TurnaroundTimeMetric:
    """Turnaround time measurement."""

    case_id: str
    slide_received: datetime
    ai_processing_start: datetime
    ai_processing_complete: datetime
    pathologist_review_start: datetime
    pathologist_review_complete: datetime
    report_finalized: datetime
    site_id: str

    @property
    def ai_processing_time(self) -> float:
        """AI processing time in minutes."""
        return (self.ai_processing_complete - self.ai_processing_start).total_seconds() / 60

    @property
    def pathologist_review_time(self) -> float:
        """Pathologist review time in minutes."""
        return (
            self.pathologist_review_complete - self.pathologist_review_start
        ).total_seconds() / 60

    @property
    def total_turnaround_time(self) -> float:
        """Total turnaround time in hours."""
        return (self.report_finalized - self.slide_received).total_seconds() / 3600


@dataclass
class UserSatisfactionSurvey:
    """User satisfaction survey response."""

    user_id: str
    user_role: str  # pathologist, technician, admin
    site_id: str
    timestamp: datetime
    ease_of_use: int  # 1-5 scale
    accuracy_perception: int  # 1-5 scale
    time_savings: int  # 1-5 scale
    overall_satisfaction: int  # 1-5 scale
    would_recommend: bool
    comments: str


@dataclass
class ClinicalOutcome:
    """Clinical outcome measurement."""

    case_id: str
    patient_id: str
    initial_diagnosis: str
    final_diagnosis: str
    treatment_plan: str
    outcome_status: str  # improved, stable, declined
    follow_up_months: int
    site_id: str
    timestamp: datetime


class ClinicalImpactTracker:
    """Tracks and analyzes clinical impact metrics."""

    def __init__(self, db_path: str = "data/clinical_impact.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Diagnostic accuracy table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnostic_accuracy (
                    case_id TEXT PRIMARY KEY,
                    ai_prediction TEXT,
                    pathologist_diagnosis TEXT,
                    ground_truth TEXT,
                    confidence_score REAL,
                    processing_time_seconds REAL,
                    timestamp TEXT,
                    site_id TEXT
                )
            """)

            # Turnaround time table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turnaround_times (
                    case_id TEXT PRIMARY KEY,
                    slide_received TEXT,
                    ai_processing_start TEXT,
                    ai_processing_complete TEXT,
                    pathologist_review_start TEXT,
                    pathologist_review_complete TEXT,
                    report_finalized TEXT,
                    site_id TEXT
                )
            """)

            # User satisfaction table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_satisfaction (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    user_role TEXT,
                    site_id TEXT,
                    timestamp TEXT,
                    ease_of_use INTEGER,
                    accuracy_perception INTEGER,
                    time_savings INTEGER,
                    overall_satisfaction INTEGER,
                    would_recommend BOOLEAN,
                    comments TEXT
                )
            """)

            # Clinical outcomes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clinical_outcomes (
                    case_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    initial_diagnosis TEXT,
                    final_diagnosis TEXT,
                    treatment_plan TEXT,
                    outcome_status TEXT,
                    follow_up_months INTEGER,
                    site_id TEXT,
                    timestamp TEXT
                )
            """)

    def track_diagnostic_accuracy(self, metric: DiagnosticAccuracyMetric):
        """Track diagnostic accuracy metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO diagnostic_accuracy 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.case_id,
                    metric.ai_prediction,
                    metric.pathologist_diagnosis,
                    metric.ground_truth,
                    metric.confidence_score,
                    metric.processing_time_seconds,
                    metric.timestamp.isoformat(),
                    metric.site_id,
                ),
            )

        logger.info(f"Tracked diagnostic accuracy for case {metric.case_id}")

    def track_turnaround_time(self, metric: TurnaroundTimeMetric):
        """Track turnaround time metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO turnaround_times
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.case_id,
                    metric.slide_received.isoformat(),
                    metric.ai_processing_start.isoformat(),
                    metric.ai_processing_complete.isoformat(),
                    metric.pathologist_review_start.isoformat(),
                    metric.pathologist_review_complete.isoformat(),
                    metric.report_finalized.isoformat(),
                    metric.site_id,
                ),
            )

        logger.info(f"Tracked turnaround time for case {metric.case_id}")

    def track_user_satisfaction(self, survey: UserSatisfactionSurvey):
        """Track user satisfaction survey."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_satisfaction 
                (user_id, user_role, site_id, timestamp, ease_of_use, 
                 accuracy_perception, time_savings, overall_satisfaction, 
                 would_recommend, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    survey.user_id,
                    survey.user_role,
                    survey.site_id,
                    survey.timestamp.isoformat(),
                    survey.ease_of_use,
                    survey.accuracy_perception,
                    survey.time_savings,
                    survey.overall_satisfaction,
                    survey.would_recommend,
                    survey.comments,
                ),
            )

        logger.info(f"Tracked user satisfaction survey from {survey.user_id}")

    def track_clinical_outcome(self, outcome: ClinicalOutcome):
        """Track clinical outcome."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO clinical_outcomes
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    outcome.case_id,
                    outcome.patient_id,
                    outcome.initial_diagnosis,
                    outcome.final_diagnosis,
                    outcome.treatment_plan,
                    outcome.outcome_status,
                    outcome.follow_up_months,
                    outcome.site_id,
                    outcome.timestamp.isoformat(),
                ),
            )

        logger.info(f"Tracked clinical outcome for case {outcome.case_id}")

    def calculate_diagnostic_accuracy_metrics(
        self, site_id: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Calculate diagnostic accuracy metrics."""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = """
            SELECT ai_prediction, pathologist_diagnosis, ground_truth, 
                   confidence_score, processing_time_seconds
            FROM diagnostic_accuracy 
            WHERE timestamp >= ?
        """
        params = [cutoff_date.isoformat()]

        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"error": "No data available for the specified period"}

        # Calculate accuracy metrics
        ai_correct = (df["ai_prediction"] == df["ground_truth"]).sum()
        pathologist_correct = (df["pathologist_diagnosis"] == df["ground_truth"]).sum()
        total_cases = len(df)

        # Agreement between AI and pathologist
        ai_pathologist_agreement = (df["ai_prediction"] == df["pathologist_diagnosis"]).sum()

        # Confidence analysis
        high_confidence_cases = df[df["confidence_score"] >= 0.8]
        high_conf_accuracy = (
            (high_confidence_cases["ai_prediction"] == high_confidence_cases["ground_truth"]).sum()
            / len(high_confidence_cases)
            if len(high_confidence_cases) > 0
            else 0
        )

        metrics = {
            "total_cases": total_cases,
            "ai_accuracy": ai_correct / total_cases,
            "pathologist_accuracy": pathologist_correct / total_cases,
            "ai_pathologist_agreement": ai_pathologist_agreement / total_cases,
            "high_confidence_accuracy": high_conf_accuracy,
            "average_confidence": df["confidence_score"].mean(),
            "average_processing_time": df["processing_time_seconds"].mean(),
            "processing_time_std": df["processing_time_seconds"].std(),
            "cases_by_confidence": {
                "high (>=0.8)": len(df[df["confidence_score"] >= 0.8]),
                "medium (0.6-0.8)": len(
                    df[(df["confidence_score"] >= 0.6) & (df["confidence_score"] < 0.8)]
                ),
                "low (<0.6)": len(df[df["confidence_score"] < 0.6]),
            },
        }

        return metrics

    def calculate_turnaround_time_metrics(
        self, site_id: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Calculate turnaround time metrics."""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = """
            SELECT slide_received, ai_processing_start, ai_processing_complete,
                   pathologist_review_start, pathologist_review_complete, report_finalized
            FROM turnaround_times 
            WHERE slide_received >= ?
        """
        params = [cutoff_date.isoformat()]

        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"error": "No data available for the specified period"}

        # Convert to datetime
        for col in df.columns:
            df[col] = pd.to_datetime(df[col])

        # Calculate time intervals
        df["ai_processing_time"] = (
            df["ai_processing_complete"] - df["ai_processing_start"]
        ).dt.total_seconds() / 60
        df["pathologist_review_time"] = (
            df["pathologist_review_complete"] - df["pathologist_review_start"]
        ).dt.total_seconds() / 60
        df["total_turnaround_time"] = (
            df["report_finalized"] - df["slide_received"]
        ).dt.total_seconds() / 3600

        metrics = {
            "total_cases": len(df),
            "ai_processing_time": {
                "mean_minutes": df["ai_processing_time"].mean(),
                "median_minutes": df["ai_processing_time"].median(),
                "std_minutes": df["ai_processing_time"].std(),
                "percentile_95": df["ai_processing_time"].quantile(0.95),
            },
            "pathologist_review_time": {
                "mean_minutes": df["pathologist_review_time"].mean(),
                "median_minutes": df["pathologist_review_time"].median(),
                "std_minutes": df["pathologist_review_time"].std(),
                "percentile_95": df["pathologist_review_time"].quantile(0.95),
            },
            "total_turnaround_time": {
                "mean_hours": df["total_turnaround_time"].mean(),
                "median_hours": df["total_turnaround_time"].median(),
                "std_hours": df["total_turnaround_time"].std(),
                "percentile_95": df["total_turnaround_time"].quantile(0.95),
            },
        }

        return metrics

    def calculate_user_satisfaction_metrics(
        self, site_id: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Calculate user satisfaction metrics."""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = """
            SELECT user_role, ease_of_use, accuracy_perception, time_savings,
                   overall_satisfaction, would_recommend
            FROM user_satisfaction 
            WHERE timestamp >= ?
        """
        params = [cutoff_date.isoformat()]

        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"error": "No data available for the specified period"}

        metrics = {
            "total_responses": len(df),
            "overall_metrics": {
                "ease_of_use": {"mean": df["ease_of_use"].mean(), "std": df["ease_of_use"].std()},
                "accuracy_perception": {
                    "mean": df["accuracy_perception"].mean(),
                    "std": df["accuracy_perception"].std(),
                },
                "time_savings": {
                    "mean": df["time_savings"].mean(),
                    "std": df["time_savings"].std(),
                },
                "overall_satisfaction": {
                    "mean": df["overall_satisfaction"].mean(),
                    "std": df["overall_satisfaction"].std(),
                },
                "recommendation_rate": df["would_recommend"].mean(),
            },
            "by_role": {},
        }

        # Calculate metrics by user role
        for role in df["user_role"].unique():
            role_df = df[df["user_role"] == role]
            metrics["by_role"][role] = {
                "count": len(role_df),
                "ease_of_use": role_df["ease_of_use"].mean(),
                "accuracy_perception": role_df["accuracy_perception"].mean(),
                "time_savings": role_df["time_savings"].mean(),
                "overall_satisfaction": role_df["overall_satisfaction"].mean(),
                "recommendation_rate": role_df["would_recommend"].mean(),
            }

        return metrics

    def perform_statistical_significance_testing(
        self, baseline_period_days: int = 90, comparison_period_days: int = 30
    ) -> Dict[str, Any]:
        """Perform statistical significance testing comparing periods."""
        baseline_end = datetime.now() - timedelta(days=comparison_period_days)
        baseline_start = baseline_end - timedelta(days=baseline_period_days)
        comparison_start = datetime.now() - timedelta(days=comparison_period_days)

        # Get diagnostic accuracy data
        with sqlite3.connect(self.db_path) as conn:
            baseline_query = """
                SELECT ai_prediction, ground_truth, processing_time_seconds
                FROM diagnostic_accuracy 
                WHERE timestamp BETWEEN ? AND ?
            """
            baseline_df = pd.read_sql_query(
                baseline_query, conn, params=[baseline_start.isoformat(), baseline_end.isoformat()]
            )

            comparison_query = """
                SELECT ai_prediction, ground_truth, processing_time_seconds
                FROM diagnostic_accuracy 
                WHERE timestamp >= ?
            """
            comparison_df = pd.read_sql_query(
                comparison_query, conn, params=[comparison_start.isoformat()]
            )

        if baseline_df.empty or comparison_df.empty:
            return {"error": "Insufficient data for statistical testing"}

        # Calculate accuracy for both periods
        baseline_accuracy = (baseline_df["ai_prediction"] == baseline_df["ground_truth"]).mean()
        comparison_accuracy = (
            comparison_df["ai_prediction"] == comparison_df["ground_truth"]
        ).mean()

        # Perform two-proportion z-test for accuracy
        baseline_correct = (baseline_df["ai_prediction"] == baseline_df["ground_truth"]).sum()
        comparison_correct = (comparison_df["ai_prediction"] == comparison_df["ground_truth"]).sum()

        # Two-sample t-test for processing times
        baseline_times = baseline_df["processing_time_seconds"]
        comparison_times = comparison_df["processing_time_seconds"]

        time_ttest = stats.ttest_ind(baseline_times, comparison_times)

        results = {
            "baseline_period": {
                "start": baseline_start.isoformat(),
                "end": baseline_end.isoformat(),
                "cases": len(baseline_df),
                "accuracy": baseline_accuracy,
                "mean_processing_time": baseline_times.mean(),
            },
            "comparison_period": {
                "start": comparison_start.isoformat(),
                "end": datetime.now().isoformat(),
                "cases": len(comparison_df),
                "accuracy": comparison_accuracy,
                "mean_processing_time": comparison_times.mean(),
            },
            "statistical_tests": {
                "accuracy_improvement": {
                    "baseline_accuracy": baseline_accuracy,
                    "comparison_accuracy": comparison_accuracy,
                    "improvement": comparison_accuracy - baseline_accuracy,
                    "improvement_percent": (
                        (comparison_accuracy - baseline_accuracy) / baseline_accuracy
                    )
                    * 100,
                },
                "processing_time_test": {
                    "t_statistic": time_ttest.statistic,
                    "p_value": time_ttest.pvalue,
                    "significant": time_ttest.pvalue < 0.05,
                    "baseline_mean": baseline_times.mean(),
                    "comparison_mean": comparison_times.mean(),
                    "improvement_seconds": baseline_times.mean() - comparison_times.mean(),
                },
            },
        }

        return results

    def calculate_cost_benefit_analysis(self, site_id: str, days_back: int = 90) -> Dict[str, Any]:
        """Calculate cost-benefit analysis for AI implementation."""
        # Get turnaround time data
        turnaround_metrics = self.calculate_turnaround_time_metrics(site_id, days_back)

        if "error" in turnaround_metrics:
            return turnaround_metrics

        # Assumptions for cost calculation
        pathologist_hourly_rate = 150  # USD per hour
        technician_hourly_rate = 25  # USD per hour
        ai_system_monthly_cost = 5000  # USD per month

        # Calculate time savings
        ai_processing_time = turnaround_metrics["ai_processing_time"]["mean_minutes"] / 60
        traditional_processing_time = 0.5  # Assume 30 minutes for manual processing
        time_saved_per_case = traditional_processing_time - ai_processing_time

        total_cases = turnaround_metrics["total_cases"]
        total_time_saved_hours = time_saved_per_case * total_cases

        # Calculate cost savings
        labor_cost_savings = total_time_saved_hours * pathologist_hourly_rate
        monthly_ai_cost = ai_system_monthly_cost * (days_back / 30)

        # Calculate ROI
        net_savings = labor_cost_savings - monthly_ai_cost
        roi_percent = (net_savings / monthly_ai_cost) * 100 if monthly_ai_cost > 0 else 0

        analysis = {
            "period_days": days_back,
            "total_cases": total_cases,
            "time_savings": {
                "minutes_per_case": time_saved_per_case * 60,
                "total_hours_saved": total_time_saved_hours,
            },
            "cost_analysis": {
                "labor_cost_savings": labor_cost_savings,
                "ai_system_cost": monthly_ai_cost,
                "net_savings": net_savings,
                "roi_percent": roi_percent,
            },
            "projections": {
                "annual_labor_savings": labor_cost_savings * (365 / days_back),
                "annual_ai_cost": ai_system_monthly_cost * 12,
                "annual_net_savings": (labor_cost_savings * (365 / days_back))
                - (ai_system_monthly_cost * 12),
                "payback_period_months": (
                    monthly_ai_cost / (labor_cost_savings * (30 / days_back))
                    if labor_cost_savings > 0
                    else float("inf")
                ),
            },
        }

        return analysis

    def generate_impact_report(
        self, site_id: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical impact report."""
        report = {
            "report_date": datetime.now().isoformat(),
            "site_id": site_id or "All Sites",
            "period_days": days_back,
            "diagnostic_accuracy": self.calculate_diagnostic_accuracy_metrics(site_id, days_back),
            "turnaround_times": self.calculate_turnaround_time_metrics(site_id, days_back),
            "user_satisfaction": self.calculate_user_satisfaction_metrics(site_id, days_back),
        }

        if site_id:
            report["cost_benefit_analysis"] = self.calculate_cost_benefit_analysis(
                site_id, days_back
            )

        # Add statistical significance testing if enough data
        if days_back >= 60:
            report["statistical_analysis"] = self.perform_statistical_significance_testing()

        return report


# Demo data generation for testing
def generate_demo_data(tracker: ClinicalImpactTracker, days_back: int = 30):
    """Generate demo data for testing."""
    import random

    sites = ["AMC001", "CH001", "RMC001"]
    diagnoses = ["Benign", "Malignant", "Atypical", "Inflammatory"]

    base_date = datetime.now() - timedelta(days=days_back)

    for i in range(500):  # Generate 500 demo cases
        case_date = base_date + timedelta(days=random.randint(0, days_back))
        site_id = random.choice(sites)
        ground_truth = random.choice(diagnoses)

        # AI prediction (90% accuracy)
        ai_prediction = ground_truth if random.random() < 0.90 else random.choice(diagnoses)

        # Pathologist diagnosis (95% accuracy)
        pathologist_diagnosis = ground_truth if random.random() < 0.95 else random.choice(diagnoses)

        # Diagnostic accuracy metric
        accuracy_metric = DiagnosticAccuracyMetric(
            case_id=f"CASE_{i:04d}",
            ai_prediction=ai_prediction,
            pathologist_diagnosis=pathologist_diagnosis,
            ground_truth=ground_truth,
            confidence_score=random.uniform(0.6, 0.99),
            processing_time_seconds=random.uniform(15, 45),
            timestamp=case_date,
            site_id=site_id,
        )
        tracker.track_diagnostic_accuracy(accuracy_metric)

        # Turnaround time metric
        slide_received = case_date
        ai_start = slide_received + timedelta(minutes=random.randint(5, 30))
        ai_complete = ai_start + timedelta(seconds=accuracy_metric.processing_time_seconds)
        path_start = ai_complete + timedelta(minutes=random.randint(10, 60))
        path_complete = path_start + timedelta(minutes=random.randint(15, 45))
        report_final = path_complete + timedelta(minutes=random.randint(5, 15))

        turnaround_metric = TurnaroundTimeMetric(
            case_id=f"CASE_{i:04d}",
            slide_received=slide_received,
            ai_processing_start=ai_start,
            ai_processing_complete=ai_complete,
            pathologist_review_start=path_start,
            pathologist_review_complete=path_complete,
            report_finalized=report_final,
            site_id=site_id,
        )
        tracker.track_turnaround_time(turnaround_metric)

    # Generate user satisfaction surveys
    users = [
        ("PATH001", "pathologist"),
        ("PATH002", "pathologist"),
        ("TECH001", "technician"),
        ("TECH002", "technician"),
        ("ADMIN001", "admin"),
    ]

    for user_id, role in users:
        for site_id in sites:
            survey = UserSatisfactionSurvey(
                user_id=user_id,
                user_role=role,
                site_id=site_id,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 30)),
                ease_of_use=random.randint(3, 5),
                accuracy_perception=random.randint(4, 5),
                time_savings=random.randint(3, 5),
                overall_satisfaction=random.randint(3, 5),
                would_recommend=random.random() > 0.1,
                comments="System working well overall",
            )
            tracker.track_user_satisfaction(survey)


if __name__ == "__main__":
    # Demo usage
    tracker = ClinicalImpactTracker()

    # Generate demo data
    generate_demo_data(tracker, days_back=60)

    # Generate impact report
    report = tracker.generate_impact_report(days_back=30)

    print("Clinical Impact Report:")
    print(f"Total cases: {report['diagnostic_accuracy']['total_cases']}")
    print(f"AI accuracy: {report['diagnostic_accuracy']['ai_accuracy']:.3f}")
    print(
        f"Average processing time: {report['diagnostic_accuracy']['average_processing_time']:.1f}s"
    )
    print(
        f"User satisfaction: {report['user_satisfaction']['overall_metrics']['overall_satisfaction']['mean']:.2f}/5"
    )
