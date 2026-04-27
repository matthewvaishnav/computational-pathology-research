"""
Continuous Learning Infrastructure for Medical AI Revolution
Phase 3: Active Learning, Federated Learning, and Model Drift Detection
"""

from .active_learning import (
    ActiveLearningSystem,
    UncertaintyBasedSampler,
    AnnotationTask,
    ExpertAnnotation,
    AnnotationQueue,
    CaseForReview
)

from .federated_learning import (
    FederatedLearningCoordinator,
    HospitalClient,
    FederatedRoundResult,
    ModelUpdate,
    PrivacyBudget,
    SecureAggregator
)

# Model drift detection - to be implemented in Task 3.3
# from .model_drift import (
#     ModelDriftDetector,
#     DriftMetrics,
#     RetrainingPipeline,
#     ABTestingFramework,
#     PerformanceMonitor
# )

__all__ = [
    # Active Learning
    "ActiveLearningSystem",
    "UncertaintyBasedSampler", 
    "AnnotationTask",
    "ExpertAnnotation",
    "AnnotationQueue",
    "CaseForReview",
    
    # Federated Learning
    "FederatedLearningCoordinator",
    "HospitalClient",
    "FederatedRoundResult", 
    "ModelUpdate",
    "PrivacyBudget",
    "SecureAggregator",
    
    # Model Drift Detection (to be implemented)
    # "ModelDriftDetector",
    # "DriftMetrics",
    # "RetrainingPipeline",
    # "ABTestingFramework",
    # "PerformanceMonitor"
]