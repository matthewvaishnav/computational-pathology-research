"""Feature importance calculation for clinical data.

Supports permutation importance, SHAP values, and gradient-based attribution
for understanding which clinical features contribute most to model predictions.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance scores with metadata."""

    feature_names: List[str]
    importance_scores: np.ndarray
    method: str
    confidence_intervals: Optional[np.ndarray] = None  # [n_features, 2] (lower, upper)


class FeatureImportanceCalculator:
    """Calculate feature importance for clinical data.

    Supports multiple methods:
    - Permutation importance: Measures performance drop when feature is shuffled
    - SHAP values: Shapley Additive exPlanations (requires shap library)
    - Gradient-based: Uses gradients w.r.t. input features

    Attributes:
        model: Trained model
        method: Importance calculation method
        device: Device for computation
        feature_names: Names of input features

    Examples:
        >>> calculator = FeatureImportanceCalculator(model, method='permutation')
        >>> importance = calculator.compute_importance(X, y, feature_names)
        >>> ranked = calculator.rank_features(importance)
    """

    def __init__(self, model: nn.Module, method: str = "permutation", device: Optional[str] = None):
        """Initialize feature importance calculator.

        Args:
            model: Trained model
            method: Importance method ('permutation', 'shap', 'gradient')
            device: Device for computation ('cuda', 'cpu', or None for auto)

        Raises:
            ValueError: If method is invalid
        """
        valid_methods = ["permutation", "shap", "gradient"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        self.model = model
        self.method = method
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"FeatureImportanceCalculator initialized with {method} method on {self.device}"
        )

    def compute_importance(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        feature_names: List[str],
        n_repeats: int = 10,
        metric: Optional[Callable] = None,
    ) -> FeatureImportance:
        """Compute feature importance scores.

        Args:
            X: Input features [N, n_features]
            y: Target labels [N]
            feature_names: Names of features
            n_repeats: Number of permutation repeats (for permutation method)
            metric: Evaluation metric (default: accuracy)

        Returns:
            FeatureImportance object with scores

        Examples:
            >>> importance = calculator.compute_importance(X, y, ['age', 'gender', 'stage'])
        """
        # Convert to tensors
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        X = X.to(self.device)
        y = y.to(self.device)

        # Validate feature names
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) != "
                f"number of features ({X.shape[1]})"
            )

        # Compute importance based on method
        if self.method == "permutation":
            scores = self.compute_permutation_importance(X, y, n_repeats, metric)
        elif self.method == "shap":
            scores = self.compute_shap_values(X, y)
        elif self.method == "gradient":
            scores = self.compute_gradient_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Normalize scores to [0, 1] with sum=1.0
        scores = self._normalize_scores(scores)

        return FeatureImportance(
            feature_names=feature_names, importance_scores=scores, method=self.method
        )

    def compute_permutation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_repeats: int = 10,
        metric: Optional[Callable] = None,
    ) -> np.ndarray:
        """Compute permutation importance.

        Args:
            X: Input features [N, n_features]
            y: Target labels [N]
            n_repeats: Number of permutation repeats
            metric: Evaluation metric (default: accuracy)

        Returns:
            Importance scores [n_features]
        """
        if metric is None:
            metric = self._accuracy_metric

        n_features = X.shape[1]

        # Baseline performance
        with torch.no_grad():
            baseline_score = metric(self.model, X, y)

        # Permutation importance for each feature
        importance_scores = np.zeros(n_features)

        for feature_idx in range(n_features):
            feature_scores = []

            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.clone()
                perm_indices = torch.randperm(X.shape[0])
                X_permuted[:, feature_idx] = X[perm_indices, feature_idx]

                # Compute performance drop
                with torch.no_grad():
                    permuted_score = metric(self.model, X_permuted, y)

                feature_scores.append(baseline_score - permuted_score)

            # Average over repeats
            importance_scores[feature_idx] = np.mean(feature_scores)

        # Ensure non-negative
        importance_scores = np.maximum(importance_scores, 0)

        logger.info(f"Computed permutation importance with {n_repeats} repeats")

        return importance_scores

    def compute_shap_values(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Compute SHAP values for feature importance.

        Args:
            X: Input features [N, n_features]
            y: Target labels [N]

        Returns:
            Importance scores [n_features]

        Note:
            Requires shap library. Falls back to gradient method if unavailable.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP library not available. Falling back to gradient method.")
            return self.compute_gradient_importance(X, y)

        # Convert to numpy for SHAP
        X_np = X.cpu().numpy()

        # Create SHAP explainer
        def model_predict(x):
            x_tensor = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
                if outputs.ndim == 2:
                    # Classification: return probabilities
                    probs = torch.softmax(outputs, dim=1)
                    return probs.cpu().numpy()
                else:
                    # Regression: return predictions
                    return outputs.cpu().numpy()

        # Use KernelExplainer (model-agnostic)
        background = shap.sample(X_np, min(100, len(X_np)))
        explainer = shap.KernelExplainer(model_predict, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_np[: min(100, len(X_np))])

        # Average absolute SHAP values across samples
        if isinstance(shap_values, list):
            # Multi-class: average over classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        importance_scores = np.mean(shap_values, axis=0)

        logger.info("Computed SHAP values")

        return importance_scores

    def compute_gradient_importance(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Compute gradient-based feature importance.

        Args:
            X: Input features [N, n_features]
            y: Target labels [N]

        Returns:
            Importance scores [n_features]
        """
        X.requires_grad = True

        # Forward pass
        outputs = self.model(X)

        # Compute loss
        if outputs.ndim == 2:
            # Classification
            loss = torch.nn.functional.cross_entropy(outputs, y)
        else:
            # Regression
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), y.float())

        # Backward pass
        loss.backward()

        # Gradient magnitude as importance
        gradients = X.grad.detach().cpu().numpy()
        importance_scores = np.mean(np.abs(gradients), axis=0)

        logger.info("Computed gradient-based importance")

        return importance_scores

    def rank_features(
        self, importance: FeatureImportance, top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """Rank features by importance scores.

        Args:
            importance: FeatureImportance object
            top_k: Return only top k features (None for all)

        Returns:
            DataFrame with ranked features

        Examples:
            >>> ranked = calculator.rank_features(importance, top_k=10)
            >>> print(ranked[['feature', 'importance']])
        """
        # Create DataFrame
        df = pd.DataFrame(
            {"feature": importance.feature_names, "importance": importance.importance_scores}
        )

        # Add confidence intervals if available
        if importance.confidence_intervals is not None:
            df["ci_lower"] = importance.confidence_intervals[:, 0]
            df["ci_upper"] = importance.confidence_intervals[:, 1]

        # Sort by importance (descending)
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        # Select top k
        if top_k is not None:
            df = df.head(top_k)

        return df

    def compute_confidence_intervals(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        feature_names: List[str],
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        n_repeats: int = 10,
    ) -> FeatureImportance:
        """Compute feature importance with confidence intervals using bootstrap.

        Args:
            X: Input features [N, n_features]
            y: Target labels [N]
            feature_names: Names of features
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95)
            n_repeats: Number of permutation repeats (for permutation method)

        Returns:
            FeatureImportance object with confidence intervals

        Examples:
            >>> importance = calculator.compute_confidence_intervals(X, y, feature_names)
        """
        # Convert to tensors
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        X = X.to(self.device)
        y = y.to(self.device)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Bootstrap sampling
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = torch.randint(0, n_samples, (n_samples,), device=self.device)
            X_boot = X[indices]
            y_boot = y[indices]

            # Compute importance
            importance = self.compute_importance(X_boot, y_boot, feature_names, n_repeats=n_repeats)
            bootstrap_scores.append(importance.importance_scores)

        bootstrap_scores = np.array(bootstrap_scores)  # [n_bootstrap, n_features]

        # Compute mean and confidence intervals
        mean_scores = np.mean(bootstrap_scores, axis=0)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_scores, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile, axis=0)

        confidence_intervals = np.stack([ci_lower, ci_upper], axis=1)  # [n_features, 2]

        # Normalize mean scores
        mean_scores = self._normalize_scores(mean_scores)

        logger.info(
            f"Computed confidence intervals with {n_bootstrap} bootstrap samples "
            f"at {confidence_level*100}% confidence level"
        )

        return FeatureImportance(
            feature_names=feature_names,
            importance_scores=mean_scores,
            method=self.method,
            confidence_intervals=confidence_intervals,
        )

    def visualize_importance(
        self,
        importance: FeatureImportance,
        output_path: Optional[Path] = None,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[Path]:
        """Visualize feature importance as bar plot.

        Args:
            importance: FeatureImportance object
            output_path: Output file path (None to display only)
            top_k: Number of top features to display
            figsize: Figure size

        Returns:
            Path to saved figure (if output_path provided)

        Examples:
            >>> calculator.visualize_importance(importance, Path('importance.png'))
        """
        # Rank features
        ranked = self.rank_features(importance, top_k=top_k)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Bar plot
        y_pos = np.arange(len(ranked))
        ax.barh(y_pos, ranked["importance"], align="center")

        # Add confidence intervals if available
        if "ci_lower" in ranked.columns:
            xerr_lower = ranked["importance"] - ranked["ci_lower"]
            xerr_upper = ranked["ci_upper"] - ranked["importance"]
            xerr = np.array([xerr_lower, xerr_upper])
            ax.errorbar(
                ranked["importance"], y_pos, xerr=xerr, fmt="none", ecolor="black", capsize=3
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranked["feature"])
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance ({importance.method.capitalize()} Method)")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature importance visualization to {output_path}")
            plt.close(fig)
            return output_path
        else:
            plt.show()
            return None

    def export_importance_scores(self, importance: FeatureImportance, output_path: Path) -> Path:
        """Export feature importance scores to CSV.

        Args:
            importance: FeatureImportance object
            output_path: Output CSV file path

        Returns:
            Path to saved CSV file

        Examples:
            >>> calculator.export_importance_scores(importance, Path('importance.csv'))
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "feature": importance.feature_names,
                "importance": importance.importance_scores,
                "method": importance.method,
            }
        )

        # Add confidence intervals if available
        if importance.confidence_intervals is not None:
            df["ci_lower"] = importance.confidence_intervals[:, 0]
            df["ci_upper"] = importance.confidence_intervals[:, 1]

        # Sort by importance
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        # Save to CSV
        df.to_csv(output_path, index=False)

        logger.info(f"Exported feature importance scores to {output_path}")

        return output_path

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] with sum=1.0.

        Args:
            scores: Raw importance scores

        Returns:
            Normalized scores
        """
        # Ensure non-negative
        scores = np.maximum(scores, 0)

        # Normalize to sum=1.0
        total = scores.sum()
        if total > 0:
            scores = scores / total
        else:
            # All zeros - uniform distribution
            scores = np.ones_like(scores) / len(scores)

        return scores

    @staticmethod
    def _accuracy_metric(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy metric.

        Args:
            model: Model
            X: Input features
            y: Target labels

        Returns:
            Accuracy score
        """
        with torch.no_grad():
            outputs = model(X)
            if outputs.ndim == 2:
                predictions = outputs.argmax(dim=1)
            else:
                predictions = (outputs.squeeze() > 0.5).long()

            accuracy = (predictions == y).float().mean().item()

        return accuracy
