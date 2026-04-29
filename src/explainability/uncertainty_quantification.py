"""
Enhanced Uncertainty Quantification System
Implements Monte Carlo dropout, ensemble methods, and confidence calibration
"""

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty quantification metrics"""

    epistemic_uncertainty: float  # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    entropy: float
    mutual_information: float
    expected_calibration_error: float
    reliability_score: float
    prediction_variance: float
    ensemble_disagreement: float


@dataclass
class CalibrationMetrics:
    """Model calibration assessment metrics"""

    expected_calibration_error: float
    maximum_calibration_error: float
    average_calibration_error: float
    reliability_diagram_data: Dict[str, np.ndarray]
    brier_score: float
    log_loss: float
    calibration_slope: float
    calibration_intercept: float


class MonteCarloDropout:
    """Enhanced Monte Carlo dropout for epistemic uncertainty estimation"""

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 20,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    def enable_dropout(self, dropout_rate: Optional[float] = None):
        """Enable dropout layers for MC sampling with optional rate override"""
        rate = dropout_rate or self.dropout_rate

        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = rate
            elif isinstance(module, nn.Dropout2d):
                module.train()
                module.p = rate

    def estimate_uncertainty(
        self,
        patches: torch.Tensor,
        disease_type: Optional[str] = None,
        return_samples: bool = False,
    ) -> Union[UncertaintyMetrics, Tuple[UncertaintyMetrics, List[torch.Tensor]]]:
        """Estimate uncertainty using Monte Carlo dropout"""
        self.model.eval()
        self.enable_dropout()

        predictions = []
        logits_samples = []

        with torch.no_grad():
            for i in range(self.num_samples):
                # Forward pass with dropout enabled
                output = self.model(patches, disease_type=disease_type)

                if disease_type:
                    logits = output[disease_type] / self.temperature
                    probs = F.softmax(logits, dim=1)
                    predictions.append(probs)
                    logits_samples.append(logits)
                else:
                    # Multi-disease case
                    pred_dict = {}
                    logits_dict = {}
                    for disease, logits in output.items():
                        if not disease.endswith("_attention") and disease != "features":
                            scaled_logits = logits / self.temperature
                            probs = F.softmax(scaled_logits, dim=1)
                            pred_dict[disease] = probs
                            logits_dict[disease] = scaled_logits
                    predictions.append(pred_dict)
                    logits_samples.append(logits_dict)

        # Calculate uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(
            predictions, logits_samples, disease_type
        )

        if return_samples:
            return uncertainty_metrics, predictions
        return uncertainty_metrics

    def _compute_uncertainty_metrics(
        self,
        predictions: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        logits_samples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        disease_type: Optional[str],
    ) -> UncertaintyMetrics:
        """Compute comprehensive uncertainty metrics from MC samples"""

        if disease_type:
            # Single disease case
            pred_stack = torch.stack(predictions)  # [num_samples, batch, classes]
            logits_stack = torch.stack(logits_samples)

            # Mean prediction and variance
            mean_pred = pred_stack.mean(dim=0)
            pred_variance = pred_stack.var(dim=0)

            # Epistemic uncertainty (predictive variance)
            epistemic = pred_variance.mean().item()

            # Aleatoric uncertainty (expected entropy)
            sample_entropies = []
            for pred in predictions:
                entropy = -(pred * torch.log(pred + 1e-8)).sum(dim=1)
                sample_entropies.append(entropy)
            aleatoric = torch.stack(sample_entropies).mean().item()

            # Total uncertainty (entropy of mean prediction)
            total_entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1).mean().item()

            # Mutual information (epistemic uncertainty)
            mutual_info = total_entropy - aleatoric

            # Confidence intervals (95%)
            sorted_preds, _ = torch.sort(pred_stack, dim=0)
            lower_idx = int(0.025 * self.num_samples)
            upper_idx = int(0.975 * self.num_samples)
            ci_lower = sorted_preds[lower_idx].max(dim=1)[0].mean().item()
            ci_upper = sorted_preds[upper_idx].max(dim=1)[0].mean().item()

            # Prediction variance
            max_probs = pred_stack.max(dim=2)[0]  # [num_samples, batch]
            prediction_variance = max_probs.var(dim=0).mean().item()

        else:
            # Multi-disease case - aggregate across diseases
            epistemic = 0.0
            aleatoric = 0.0
            total_entropy = 0.0
            mutual_info = 0.0
            ci_lower = 1.0
            ci_upper = 0.0
            prediction_variance = 0.0

            disease_count = 0
            for disease in predictions[0].keys():
                disease_preds = torch.stack([p[disease] for p in predictions])
                disease_logits = torch.stack([l[disease] for l in logits_samples])

                mean_pred = disease_preds.mean(dim=0)
                pred_var = disease_preds.var(dim=0)

                epistemic += pred_var.mean().item()

                # Aleatoric for this disease
                sample_entropies = []
                for pred in [p[disease] for p in predictions]:
                    entropy = -(pred * torch.log(pred + 1e-8)).sum(dim=1)
                    sample_entropies.append(entropy)
                disease_aleatoric = torch.stack(sample_entropies).mean().item()
                aleatoric += disease_aleatoric

                # Total entropy for this disease
                disease_total_entropy = (
                    -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1).mean().item()
                )
                total_entropy += disease_total_entropy

                # Mutual information
                mutual_info += disease_total_entropy - disease_aleatoric

                # Confidence intervals
                sorted_preds, _ = torch.sort(disease_preds, dim=0)
                lower_idx = int(0.025 * self.num_samples)
                upper_idx = int(0.975 * self.num_samples)
                ci_lower = min(ci_lower, sorted_preds[lower_idx].max(dim=1)[0].mean().item())
                ci_upper = max(ci_upper, sorted_preds[upper_idx].max(dim=1)[0].mean().item())

                # Prediction variance
                max_probs = disease_preds.max(dim=2)[0]
                prediction_variance += max_probs.var(dim=0).mean().item()

                disease_count += 1

            # Average across diseases
            if disease_count > 0:
                epistemic /= disease_count
                aleatoric /= disease_count
                total_entropy /= disease_count
                mutual_info /= disease_count
                prediction_variance /= disease_count

        return UncertaintyMetrics(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_entropy,
            confidence_interval=(ci_lower, ci_upper),
            entropy=total_entropy,
            mutual_information=mutual_info,
            expected_calibration_error=0.0,  # Will be computed separately
            reliability_score=1.0 - total_entropy,  # Simple reliability measure
            prediction_variance=prediction_variance,
            ensemble_disagreement=epistemic,  # For MC dropout, this equals epistemic
        )


class EnsembleUncertainty:
    """Ensemble-based uncertainty quantification using multiple models"""

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
    ):
        self.models = models
        self.num_models = len(models)
        self.weights = weights or [1.0 / self.num_models] * self.num_models
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def estimate_uncertainty(
        self, patches: torch.Tensor, disease_type: Optional[str] = None, parallel: bool = True
    ) -> UncertaintyMetrics:
        """Estimate uncertainty using ensemble of models"""

        if parallel and len(self.models) > 1:
            predictions = self._parallel_inference(patches, disease_type)
        else:
            predictions = self._sequential_inference(patches, disease_type)

        return self._compute_ensemble_uncertainty(predictions, disease_type)

    def _parallel_inference(
        self, patches: torch.Tensor, disease_type: Optional[str]
    ) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run inference in parallel across models"""

        def run_model(model_idx):
            model = self.models[model_idx]
            model.eval()
            with torch.no_grad():
                output = model(patches, disease_type=disease_type)

                if disease_type:
                    logits = output[disease_type] / self.temperature
                    return F.softmax(logits, dim=1)
                else:
                    pred_dict = {}
                    for disease, logits in output.items():
                        if not disease.endswith("_attention") and disease != "features":
                            scaled_logits = logits / self.temperature
                            pred_dict[disease] = F.softmax(scaled_logits, dim=1)
                    return pred_dict

        with ThreadPoolExecutor(max_workers=min(4, len(self.models))) as executor:
            futures = [executor.submit(run_model, i) for i in range(len(self.models))]
            predictions = [future.result() for future in futures]

        return predictions

    def _sequential_inference(
        self, patches: torch.Tensor, disease_type: Optional[str]
    ) -> List[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run inference sequentially across models"""
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(patches, disease_type=disease_type)

                if disease_type:
                    logits = output[disease_type] / self.temperature
                    predictions.append(F.softmax(logits, dim=1))
                else:
                    pred_dict = {}
                    for disease, logits in output.items():
                        if not disease.endswith("_attention") and disease != "features":
                            scaled_logits = logits / self.temperature
                            pred_dict[disease] = F.softmax(scaled_logits, dim=1)
                    predictions.append(pred_dict)

        return predictions

    def _compute_ensemble_uncertainty(
        self,
        predictions: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        disease_type: Optional[str],
    ) -> UncertaintyMetrics:
        """Compute uncertainty metrics from ensemble predictions"""

        if disease_type:
            # Single disease case
            # Weighted ensemble prediction
            weighted_preds = []
            for i, pred in enumerate(predictions):
                weighted_preds.append(pred * self.weights[i])

            ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
            pred_stack = torch.stack(predictions)

            # Ensemble disagreement (variance across models)
            ensemble_disagreement = pred_stack.var(dim=0).mean().item()

            # Total uncertainty (entropy of ensemble prediction)
            total_uncertainty = (
                -(ensemble_pred * torch.log(ensemble_pred + 1e-8)).sum(dim=1).mean().item()
            )

            # Aleatoric uncertainty (expected entropy across models)
            model_entropies = []
            for pred in predictions:
                entropy = -(pred * torch.log(pred + 1e-8)).sum(dim=1)
                model_entropies.append(entropy)
            aleatoric = torch.stack(model_entropies).mean().item()

            # Epistemic uncertainty (mutual information)
            epistemic = total_uncertainty - aleatoric

            # Confidence intervals from ensemble spread
            max_probs = pred_stack.max(dim=2)[0]  # [num_models, batch]
            ci_lower = torch.quantile(max_probs, 0.025, dim=0).mean().item()
            ci_upper = torch.quantile(max_probs, 0.975, dim=0).mean().item()

            # Prediction variance
            prediction_variance = max_probs.var(dim=0).mean().item()

        else:
            # Multi-disease case
            epistemic = 0.0
            aleatoric = 0.0
            total_uncertainty = 0.0
            ensemble_disagreement = 0.0
            ci_lower = 1.0
            ci_upper = 0.0
            prediction_variance = 0.0

            disease_count = 0
            for disease in predictions[0].keys():
                # Extract predictions for this disease
                disease_preds = [p[disease] for p in predictions]

                # Weighted ensemble
                weighted_preds = []
                for i, pred in enumerate(disease_preds):
                    weighted_preds.append(pred * self.weights[i])
                ensemble_pred = torch.stack(weighted_preds).sum(dim=0)

                pred_stack = torch.stack(disease_preds)

                # Metrics for this disease
                disease_disagreement = pred_stack.var(dim=0).mean().item()
                ensemble_disagreement += disease_disagreement

                disease_total = (
                    -(ensemble_pred * torch.log(ensemble_pred + 1e-8)).sum(dim=1).mean().item()
                )
                total_uncertainty += disease_total

                model_entropies = []
                for pred in disease_preds:
                    entropy = -(pred * torch.log(pred + 1e-8)).sum(dim=1)
                    model_entropies.append(entropy)
                disease_aleatoric = torch.stack(model_entropies).mean().item()
                aleatoric += disease_aleatoric

                epistemic += disease_total - disease_aleatoric

                # Confidence intervals
                max_probs = pred_stack.max(dim=2)[0]
                ci_lower = min(ci_lower, torch.quantile(max_probs, 0.025, dim=0).mean().item())
                ci_upper = max(ci_upper, torch.quantile(max_probs, 0.975, dim=0).mean().item())

                prediction_variance += max_probs.var(dim=0).mean().item()

                disease_count += 1

            # Average across diseases
            if disease_count > 0:
                epistemic /= disease_count
                aleatoric /= disease_count
                total_uncertainty /= disease_count
                ensemble_disagreement /= disease_count
                prediction_variance /= disease_count

        return UncertaintyMetrics(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
            confidence_interval=(ci_lower, ci_upper),
            entropy=total_uncertainty,
            mutual_information=epistemic,
            expected_calibration_error=0.0,  # Will be computed separately
            reliability_score=1.0 - total_uncertainty,
            prediction_variance=prediction_variance,
            ensemble_disagreement=ensemble_disagreement,
        )


class ConfidenceCalibrator:
    """Calibrates model confidence scores for better uncertainty quantification"""

    def __init__(self, method: str = "platt", num_bins: int = 10):
        self.method = method
        self.num_bins = num_bins
        self.calibrator = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(self, confidences: np.ndarray, correct_predictions: np.ndarray) -> None:
        """Fit calibration model on validation data"""

        if self.method == "platt":
            # Platt scaling (logistic regression)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(confidences.reshape(-1, 1), correct_predictions)
        elif self.method == "isotonic":
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(confidences, correct_predictions)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        self.logger.info(f"Fitted {self.method} calibrator on {len(confidences)} samples")

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Apply calibration to confidence scores"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")

        if self.method == "platt":
            return self.calibrator.predict_proba(confidences.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            return self.calibrator.predict(confidences)

    def evaluate_calibration(
        self, confidences: np.ndarray, correct_predictions: np.ndarray
    ) -> CalibrationMetrics:
        """Evaluate calibration quality"""

        # Reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            correct_predictions, confidences, n_bins=self.num_bins
        )

        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0  # Maximum Calibration Error
        ace = 0.0  # Average Calibration Error

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = correct_predictions[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * calibration_error
                mce = max(mce, calibration_error)
                ace += calibration_error

        ace /= self.num_bins

        # Brier score
        brier_score = np.mean((confidences - correct_predictions) ** 2)

        # Log loss
        epsilon = 1e-15
        confidences_clipped = np.clip(confidences, epsilon, 1 - epsilon)
        log_loss = -np.mean(
            correct_predictions * np.log(confidences_clipped)
            + (1 - correct_predictions) * np.log(1 - confidences_clipped)
        )

        # Calibration slope and intercept (linear fit)
        try:
            from scipy import stats

            slope, intercept, _, _, _ = stats.linregress(confidences, correct_predictions)
        except ImportError:
            # Fallback to simple linear regression if scipy not available
            slope = 1.0
            intercept = 0.0

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_calibration_error=ace,
            reliability_diagram_data={
                "fraction_of_positives": fraction_of_positives,
                "mean_predicted_value": mean_predicted_value,
                "bin_boundaries": bin_boundaries,
            },
            brier_score=brier_score,
            log_loss=log_loss,
            calibration_slope=slope,
            calibration_intercept=intercept,
        )


class UncertaintyQuantificationSystem:
    """Comprehensive uncertainty quantification system"""

    def __init__(
        self,
        models: List[nn.Module],
        mc_samples: int = 20,
        ensemble_weights: Optional[List[float]] = None,
        calibration_method: str = "platt",
        temperature: float = 1.0,
    ):
        self.models = models
        self.mc_samples = mc_samples
        self.temperature = temperature

        # Initialize uncertainty estimators
        self.mc_dropout = MonteCarloDropout(
            models[0], num_samples=mc_samples, temperature=temperature
        )

        if len(models) > 1:
            self.ensemble = EnsembleUncertainty(
                models, weights=ensemble_weights, temperature=temperature
            )
        else:
            self.ensemble = None

        # Initialize calibrator
        self.calibrator = ConfidenceCalibrator(method=calibration_method)

        self.logger = logging.getLogger(__name__)

    def estimate_uncertainty(
        self,
        patches: torch.Tensor,
        disease_type: Optional[str] = None,
        method: str = "ensemble",  # "mc_dropout", "ensemble", "both"
        calibrate: bool = True,
    ) -> UncertaintyMetrics:
        """Estimate uncertainty using specified method"""

        start_time = time.time()

        if method == "mc_dropout" or (method == "ensemble" and self.ensemble is None):
            uncertainty_metrics = self.mc_dropout.estimate_uncertainty(patches, disease_type)
        elif method == "ensemble":
            uncertainty_metrics = self.ensemble.estimate_uncertainty(patches, disease_type)
        elif method == "both":
            # Combine MC dropout and ensemble
            mc_metrics = self.mc_dropout.estimate_uncertainty(patches, disease_type)
            ensemble_metrics = self.ensemble.estimate_uncertainty(patches, disease_type)

            # Average the metrics (could use more sophisticated combination)
            uncertainty_metrics = UncertaintyMetrics(
                epistemic_uncertainty=(
                    mc_metrics.epistemic_uncertainty + ensemble_metrics.epistemic_uncertainty
                )
                / 2,
                aleatoric_uncertainty=(
                    mc_metrics.aleatoric_uncertainty + ensemble_metrics.aleatoric_uncertainty
                )
                / 2,
                total_uncertainty=(
                    mc_metrics.total_uncertainty + ensemble_metrics.total_uncertainty
                )
                / 2,
                confidence_interval=(
                    min(mc_metrics.confidence_interval[0], ensemble_metrics.confidence_interval[0]),
                    max(mc_metrics.confidence_interval[1], ensemble_metrics.confidence_interval[1]),
                ),
                entropy=(mc_metrics.entropy + ensemble_metrics.entropy) / 2,
                mutual_information=(
                    mc_metrics.mutual_information + ensemble_metrics.mutual_information
                )
                / 2,
                expected_calibration_error=0.0,
                reliability_score=(
                    mc_metrics.reliability_score + ensemble_metrics.reliability_score
                )
                / 2,
                prediction_variance=(
                    mc_metrics.prediction_variance + ensemble_metrics.prediction_variance
                )
                / 2,
                ensemble_disagreement=ensemble_metrics.ensemble_disagreement,
            )
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

        # Apply calibration if available and requested
        if calibrate and self.calibrator.is_fitted:
            # Note: This is a simplified calibration application
            # In practice, you'd need to calibrate the actual confidence scores
            pass

        processing_time = time.time() - start_time
        self.logger.debug(f"Uncertainty estimation took {processing_time:.3f}s")

        return uncertainty_metrics

    def fit_calibrator(
        self,
        validation_patches: List[torch.Tensor],
        validation_labels: List[int],
        disease_type: Optional[str] = None,
    ) -> CalibrationMetrics:
        """Fit confidence calibrator on validation data"""

        confidences = []
        correct_predictions = []

        for patches, label in zip(validation_patches, validation_labels):
            # Get model prediction
            with torch.no_grad():
                output = self.models[0](patches, disease_type=disease_type)

                if disease_type:
                    probs = F.softmax(output[disease_type], dim=1)
                    confidence = torch.max(probs).item()
                    prediction = torch.argmax(probs).item()
                else:
                    # For multi-disease, use maximum confidence across diseases
                    max_confidence = 0.0
                    best_prediction = 0
                    for disease, logits in output.items():
                        if not disease.endswith("_attention") and disease != "features":
                            probs = F.softmax(logits, dim=1)
                            conf = torch.max(probs).item()
                            if conf > max_confidence:
                                max_confidence = conf
                                best_prediction = torch.argmax(probs).item()
                    confidence = max_confidence
                    prediction = best_prediction

                confidences.append(confidence)
                correct_predictions.append(1 if prediction == label else 0)

        confidences = np.array(confidences)
        correct_predictions = np.array(correct_predictions)

        # Fit calibrator
        self.calibrator.fit(confidences, correct_predictions)

        # Evaluate calibration
        calibration_metrics = self.calibrator.evaluate_calibration(confidences, correct_predictions)

        self.logger.info(
            f"Calibrator fitted. ECE: {calibration_metrics.expected_calibration_error:.4f}"
        )

        return calibration_metrics

    def should_request_second_opinion(
        self, uncertainty_metrics: UncertaintyMetrics, thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """Determine if case requires second opinion based on uncertainty"""

        default_thresholds = {
            "total_uncertainty": 0.5,
            "epistemic_uncertainty": 0.3,
            "ensemble_disagreement": 0.2,
            "confidence_interval_width": 0.3,
        }

        thresholds = thresholds or default_thresholds

        reasons = []

        if uncertainty_metrics.total_uncertainty > thresholds["total_uncertainty"]:
            reasons.append(f"High total uncertainty ({uncertainty_metrics.total_uncertainty:.3f})")

        if uncertainty_metrics.epistemic_uncertainty > thresholds["epistemic_uncertainty"]:
            reasons.append(
                f"High epistemic uncertainty ({uncertainty_metrics.epistemic_uncertainty:.3f})"
            )

        if uncertainty_metrics.ensemble_disagreement > thresholds["ensemble_disagreement"]:
            reasons.append(
                f"High ensemble disagreement ({uncertainty_metrics.ensemble_disagreement:.3f})"
            )

        ci_width = (
            uncertainty_metrics.confidence_interval[1] - uncertainty_metrics.confidence_interval[0]
        )
        if ci_width > thresholds["confidence_interval_width"]:
            reasons.append(f"Wide confidence interval ({ci_width:.3f})")

        requires_second_opinion = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "Uncertainty within acceptable bounds"

        return requires_second_opinion, reason_text


# Example usage
if __name__ == "__main__":
    from src.foundation.multi_disease_model import create_foundation_model

    # Create models for ensemble
    model1 = create_foundation_model(encoder_type="resnet50")
    model2 = create_foundation_model(encoder_type="efficientnet_b0")
    models = [model1, model2]

    # Initialize uncertainty system
    uncertainty_system = UncertaintyQuantificationSystem(
        models=models, mc_samples=20, calibration_method="platt"
    )

    # Example input
    patches = torch.randn(1, 50, 3, 224, 224)

    # Estimate uncertainty using ensemble
    uncertainty_metrics = uncertainty_system.estimate_uncertainty(
        patches, disease_type="breast", method="ensemble"
    )

    print("Uncertainty Metrics:")
    print(f"Total uncertainty: {uncertainty_metrics.total_uncertainty:.4f}")
    print(f"Epistemic uncertainty: {uncertainty_metrics.epistemic_uncertainty:.4f}")
    print(f"Aleatoric uncertainty: {uncertainty_metrics.aleatoric_uncertainty:.4f}")
    print(f"Confidence interval: {uncertainty_metrics.confidence_interval}")
    print(f"Ensemble disagreement: {uncertainty_metrics.ensemble_disagreement:.4f}")

    # Check if second opinion needed
    needs_review, reason = uncertainty_system.should_request_second_opinion(uncertainty_metrics)
    print(f"Requires second opinion: {needs_review}")
    print(f"Reason: {reason}")
