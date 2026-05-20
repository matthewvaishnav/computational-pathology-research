"""
Cross-Validation Strategy for Clinical Validation

Implements sophisticated cross-validation strategies specifically designed
for medical AI validation, accounting for site effects, patient stratification,
and clinical requirements.
"""

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class ValidationStrategy(Enum):
    """Types of cross-validation strategies"""

    STRATIFIED_K_FOLD = "stratified_k_fold"
    SITE_STRATIFIED = "site_stratified"
    PATIENT_STRATIFIED = "patient_stratified"
    LEAVE_ONE_SITE_OUT = "leave_one_site_out"
    TEMPORAL_SPLIT = "temporal_split"
    NESTED_CV = "nested_cv"
    MONTE_CARLO_CV = "monte_carlo_cv"
    BOOTSTRAP = "bootstrap"


@dataclass
class ValidationConfig:
    """Configuration for cross-validation strategy"""

    strategy: ValidationStrategy
    n_folds: int = 5
    n_repeats: int = 1
    test_size: float = 0.2
    random_state: int = 42
    stratify_by: Optional[List[str]] = None
    group_by: Optional[str] = None
    min_samples_per_fold: int = 10
    balance_sites: bool = True
    ensure_disease_representation: bool = True


@dataclass
class ValidationFold:
    """Single validation fold data"""

    fold_id: int
    train_indices: List[int]
    val_indices: List[int]
    test_indices: Optional[List[int]] = None
    train_sites: List[str] = None
    val_sites: List[str] = None
    test_sites: Optional[List[str]] = None
    metadata: Dict = None


class ClinicalCrossValidator:
    """Advanced cross-validation for clinical AI validation"""

    def __init__(self, config: ValidationConfig):
        """Initialize cross-validator with configuration"""
        self.config = config
        np.random.seed(config.random_state)
        random.seed(config.random_state)

    def create_validation_splits(
        self,
        data: pd.DataFrame,
        target_column: str = "label",
        site_column: str = "site_id",
        patient_column: str = "patient_id",
    ) -> List[ValidationFold]:
        """Create validation splits based on strategy"""

        if self.config.strategy == ValidationStrategy.STRATIFIED_K_FOLD:
            return self._stratified_k_fold(data, target_column)

        elif self.config.strategy == ValidationStrategy.SITE_STRATIFIED:
            return self._site_stratified_cv(data, target_column, site_column)

        elif self.config.strategy == ValidationStrategy.PATIENT_STRATIFIED:
            return self._patient_stratified_cv(data, target_column, patient_column)

        elif self.config.strategy == ValidationStrategy.LEAVE_ONE_SITE_OUT:
            return self._leave_one_site_out(data, target_column, site_column)

        elif self.config.strategy == ValidationStrategy.TEMPORAL_SPLIT:
            return self._temporal_split(data, target_column)

        elif self.config.strategy == ValidationStrategy.NESTED_CV:
            return self._nested_cv(data, target_column, site_column, patient_column)

        elif self.config.strategy == ValidationStrategy.MONTE_CARLO_CV:
            return self._monte_carlo_cv(data, target_column, site_column)

        elif self.config.strategy == ValidationStrategy.BOOTSTRAP:
            return self._bootstrap_cv(data, target_column)

        else:
            raise ValueError(f"Unknown validation strategy: {self.config.strategy}")

    def _stratified_k_fold(self, data: pd.DataFrame, target_column: str) -> List[ValidationFold]:
        """Standard stratified k-fold cross-validation"""
        skf = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state
        )

        folds = []
        y = data[target_column].values

        for fold_id, (train_idx, val_idx) in enumerate(skf.split(data, y)):
            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                metadata={
                    "train_class_distribution": Counter(y[train_idx]),
                    "val_class_distribution": Counter(y[val_idx]),
                },
            )
            folds.append(fold)

        return folds

    def _site_stratified_cv(
        self, data: pd.DataFrame, target_column: str, site_column: str
    ) -> List[ValidationFold]:
        """Site-stratified cross-validation ensuring site balance"""
        sites = data[site_column].unique()
        n_sites = len(sites)

        if n_sites < self.config.n_folds:
            logger.warning(f"Only {n_sites} sites available for {self.config.n_folds} folds")
            self.config.n_folds = n_sites

        # Group sites into folds
        site_folds = np.array_split(sites, self.config.n_folds)

        folds = []
        for fold_id in range(self.config.n_folds):
            val_sites = site_folds[fold_id].tolist()
            train_sites = [s for s in sites if s not in val_sites]

            train_mask = data[site_column].isin(train_sites)
            val_mask = data[site_column].isin(val_sites)

            train_indices = data[train_mask].index.tolist()
            val_indices = data[val_mask].index.tolist()

            # Ensure minimum samples per fold
            if len(val_indices) < self.config.min_samples_per_fold:
                logger.warning(f"Fold {fold_id} has only {len(val_indices)} validation samples")

            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_sites=train_sites,
                val_sites=val_sites,
                metadata={
                    "n_train_sites": len(train_sites),
                    "n_val_sites": len(val_sites),
                    "train_class_distribution": Counter(data.loc[train_indices, target_column]),
                    "val_class_distribution": Counter(data.loc[val_indices, target_column]),
                },
            )
            folds.append(fold)

        return folds

    def _patient_stratified_cv(
        self, data: pd.DataFrame, target_column: str, patient_column: str
    ) -> List[ValidationFold]:
        """Patient-stratified CV ensuring no patient leakage"""
        # Group by patient and get patient-level labels
        patient_groups = data.groupby(patient_column)[target_column].first()
        patients = patient_groups.index.values
        patient_labels = patient_groups.values

        # Stratified split at patient level
        sgkf = StratifiedGroupKFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state
        )

        folds = []
        for fold_id, (train_patients_idx, val_patients_idx) in enumerate(
            sgkf.split(patients, patient_labels, groups=patients)
        ):
            train_patients = patients[train_patients_idx]
            val_patients = patients[val_patients_idx]

            train_mask = data[patient_column].isin(train_patients)
            val_mask = data[patient_column].isin(val_patients)

            train_indices = data[train_mask].index.tolist()
            val_indices = data[val_mask].index.tolist()

            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    "n_train_patients": len(train_patients),
                    "n_val_patients": len(val_patients),
                    "train_class_distribution": Counter(data.loc[train_indices, target_column]),
                    "val_class_distribution": Counter(data.loc[val_indices, target_column]),
                },
            )
            folds.append(fold)

        return folds

    def _leave_one_site_out(
        self, data: pd.DataFrame, target_column: str, site_column: str
    ) -> List[ValidationFold]:
        """Leave-one-site-out cross-validation"""
        sites = data[site_column].unique()
        folds = []

        for fold_id, test_site in enumerate(sites):
            train_sites = [s for s in sites if s != test_site]

            train_mask = data[site_column].isin(train_sites)
            test_mask = data[site_column] == test_site

            train_indices = data[train_mask].index.tolist()
            test_indices = data[test_mask].index.tolist()

            # Further split training data for validation
            train_data = data.loc[train_indices]
            if len(train_data) > 0:
                train_idx, val_idx = train_test_split(
                    train_indices,
                    test_size=0.2,
                    stratify=data.loc[train_indices, target_column],
                    random_state=self.config.random_state,
                )
            else:
                train_idx, val_idx = [], []

            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_indices,
                train_sites=[s for s in train_sites],
                val_sites=[s for s in train_sites],  # Val sites are subset of train sites
                test_sites=[test_site],
                metadata={
                    "test_site": test_site,
                    "n_train_sites": len(train_sites),
                    "test_class_distribution": Counter(data.loc[test_indices, target_column]),
                },
            )
            folds.append(fold)

        return folds

    def _temporal_split(
        self, data: pd.DataFrame, target_column: str, date_column: str = "acquisition_date"
    ) -> List[ValidationFold]:
        """Temporal split for time-series validation"""
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")

        # Sort by date
        data_sorted = data.sort_values(date_column)
        n_samples = len(data_sorted)

        folds = []
        fold_size = n_samples // self.config.n_folds

        for fold_id in range(self.config.n_folds):
            # Use all previous data for training
            train_end = (fold_id + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples)

            if fold_id == 0:
                # First fold: use first portion for training
                train_indices = data_sorted.iloc[:fold_size].index.tolist()
                val_indices = data_sorted.iloc[fold_size : 2 * fold_size].index.tolist()
            else:
                train_indices = data_sorted.iloc[:train_end].index.tolist()
                val_indices = data_sorted.iloc[val_start:val_end].index.tolist()

            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    "temporal_split": True,
                    "train_date_range": (
                        data_sorted.iloc[0][date_column],
                        data_sorted.iloc[train_end - 1][date_column],
                    ),
                    "val_date_range": (
                        data_sorted.iloc[val_start][date_column],
                        data_sorted.iloc[val_end - 1][date_column],
                    ),
                },
            )
            folds.append(fold)

        return folds

    def _nested_cv(
        self, data: pd.DataFrame, target_column: str, site_column: str, patient_column: str
    ) -> List[ValidationFold]:
        """Nested cross-validation for hyperparameter tuning"""
        # Outer loop: site-stratified
        outer_folds = self._site_stratified_cv(data, target_column, site_column)

        nested_folds = []
        for outer_fold in outer_folds:
            # Inner loop: patient-stratified on training data
            train_data = data.iloc[outer_fold.train_indices]

            inner_config = ValidationConfig(
                strategy=ValidationStrategy.PATIENT_STRATIFIED,
                n_folds=3,  # Smaller for inner loop
                random_state=self.config.random_state,
            )
            inner_validator = ClinicalCrossValidator(inner_config)
            inner_folds = inner_validator._patient_stratified_cv(
                train_data, target_column, patient_column
            )

            # Adjust indices to global indices
            for inner_fold in inner_folds:
                inner_fold.train_indices = [
                    outer_fold.train_indices[i] for i in inner_fold.train_indices
                ]
                inner_fold.val_indices = [
                    outer_fold.train_indices[i] for i in inner_fold.val_indices
                ]
                inner_fold.test_indices = outer_fold.val_indices
                inner_fold.metadata["outer_fold_id"] = outer_fold.fold_id

            nested_folds.extend(inner_folds)

        return nested_folds

    def _monte_carlo_cv(
        self, data: pd.DataFrame, target_column: str, site_column: str
    ) -> List[ValidationFold]:
        """Monte Carlo cross-validation with random splits"""
        folds = []

        for fold_id in range(self.config.n_repeats):
            # Random stratified split
            train_idx, val_idx = train_test_split(
                range(len(data)),
                test_size=1.0 / self.config.n_folds,
                stratify=data[target_column],
                random_state=self.config.random_state + fold_id,
            )

            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata={
                    "monte_carlo_iteration": fold_id,
                    "train_sites": data.iloc[train_idx][site_column].unique().tolist(),
                    "val_sites": data.iloc[val_idx][site_column].unique().tolist(),
                },
            )
            folds.append(fold)

        return folds

    def _bootstrap_cv(self, data: pd.DataFrame, target_column: str) -> List[ValidationFold]:
        """Bootstrap cross-validation"""
        folds = []
        n_samples = len(data)

        for fold_id in range(self.config.n_repeats):
            # Bootstrap sample (with replacement)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Out-of-bag samples for validation
            oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))

            fold = ValidationFold(
                fold_id=fold_id,
                train_indices=bootstrap_indices.tolist(),
                val_indices=oob_indices,
                metadata={
                    "bootstrap_iteration": fold_id,
                    "n_unique_train_samples": len(set(bootstrap_indices)),
                    "oob_fraction": len(oob_indices) / n_samples,
                },
            )
            folds.append(fold)

        return folds

    def validate_splits(
        self,
        folds: List[ValidationFold],
        data: pd.DataFrame,
        target_column: str,
        site_column: str = None,
    ) -> Dict:
        """Validate the quality of cross-validation splits"""
        validation_report = {
            "n_folds": len(folds),
            "fold_sizes": [],
            "class_balance": [],
            "site_distribution": [],
            "overlap_check": True,
        }

        all_train_indices = set()
        all_val_indices = set()

        for fold in folds:
            # Check fold sizes
            validation_report["fold_sizes"].append(
                {
                    "fold_id": fold.fold_id,
                    "train_size": len(fold.train_indices),
                    "val_size": len(fold.val_indices),
                }
            )

            # Check class balance
            train_labels = data.iloc[fold.train_indices][target_column]
            val_labels = data.iloc[fold.val_indices][target_column]

            validation_report["class_balance"].append(
                {
                    "fold_id": fold.fold_id,
                    "train_distribution": dict(Counter(train_labels)),
                    "val_distribution": dict(Counter(val_labels)),
                }
            )

            # Check site distribution
            if site_column and site_column in data.columns:
                train_sites = data.iloc[fold.train_indices][site_column]
                val_sites = data.iloc[fold.val_indices][site_column]

                validation_report["site_distribution"].append(
                    {
                        "fold_id": fold.fold_id,
                        "train_sites": dict(Counter(train_sites)),
                        "val_sites": dict(Counter(val_sites)),
                    }
                )

            # Check for overlap
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)

            if train_set & val_set:
                validation_report["overlap_check"] = False
                logger.error(f"Overlap detected in fold {fold.fold_id}")

            all_train_indices.update(train_set)
            all_val_indices.update(val_set)

        # Overall statistics
        validation_report["coverage"] = {
            "total_samples": len(data),
            "samples_in_train": len(all_train_indices),
            "samples_in_val": len(all_val_indices),
            "coverage_ratio": len(all_train_indices | all_val_indices) / len(data),
        }

        return validation_report

    def get_fold_iterator(
        self, folds: List[ValidationFold], data: pd.DataFrame
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get iterator over train/validation data for each fold"""
        for fold in folds:
            train_data = data.iloc[fold.train_indices]
            val_data = data.iloc[fold.val_indices]
            yield train_data, val_data


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame(
        {
            "patient_id": [f"P{i//5}" for i in range(n_samples)],  # 5 samples per patient
            "site_id": np.random.choice(["Site_A", "Site_B", "Site_C", "Site_D"], n_samples),
            "label": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            "acquisition_date": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        }
    )

    # Test different validation strategies
    strategies = [
        ValidationStrategy.STRATIFIED_K_FOLD,
        ValidationStrategy.SITE_STRATIFIED,
        ValidationStrategy.PATIENT_STRATIFIED,
        ValidationStrategy.LEAVE_ONE_SITE_OUT,
    ]

    for strategy in strategies:
        print(f"\nTesting {strategy.value}:")

        config = ValidationConfig(strategy=strategy, n_folds=5, random_state=42)

        validator = ClinicalCrossValidator(config)
        folds = validator.create_validation_splits(sample_data)

        # Validate splits
        report = validator.validate_splits(folds, sample_data, "label", "site_id")

        print(f"  Number of folds: {report['n_folds']}")
        print(f"  Coverage ratio: {report['coverage']['coverage_ratio']:.3f}")
        print(f"  Overlap check passed: {report['overlap_check']}")

        # Show fold sizes
        fold_sizes = [f["train_size"] + f["val_size"] for f in report["fold_sizes"]]
        print(f"  Fold sizes: {fold_sizes}")
