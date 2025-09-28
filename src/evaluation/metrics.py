"""
Evaluation metrics for emotion prediction tasks.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EmotionMetrics:
    """
    Metrics for emotion prediction evaluation.
    """

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate Pearson correlation coefficient and p-value."""
        if len(y_true) < 2:
            return 0.0, 1.0
        try:
            corr, p_value = pearsonr(y_true, y_pred)
            return corr if not np.isnan(corr) else 0.0, p_value if not np.isnan(p_value) else 1.0
        except:
            return 0.0, 1.0

    @staticmethod
    def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate Spearman correlation coefficient and p-value."""
        if len(y_true) < 2:
            return 0.0, 1.0
        try:
            corr, p_value = spearmanr(y_true, y_pred)
            return corr if not np.isnan(corr) else 0.0, p_value if not np.isnan(p_value) else 1.0
        except:
            return 0.0, 1.0

    @staticmethod
    def r2_score_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        try:
            score = r2_score(y_true, y_pred)
            return score if not np.isnan(score) else 0.0
        except:
            return 0.0

    @staticmethod
    def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Concordance Correlation Coefficient (CCC).
        Often used for emotion prediction evaluation.
        """
        if len(y_true) < 2:
            return 0.0

        try:
            # Calculate means
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)

            # Calculate variances
            var_true = np.var(y_true, ddof=1)
            var_pred = np.var(y_pred, ddof=1)

            # Calculate covariance
            cov = np.cov(y_true, y_pred, ddof=1)[0, 1]

            # Calculate CCC
            numerator = 2 * cov
            denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

            if denominator == 0:
                return 0.0

            ccc = numerator / denominator
            return ccc if not np.isnan(ccc) else 0.0
        except:
            return 0.0

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Calculate Mean Absolute Percentage Error."""
        try:
            y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
            mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
            return mape if not np.isnan(mape) else 100.0
        except:
            return 100.0

    @staticmethod
    def emotion_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Euclidean distance in emotion space.
        Useful for multi-dimensional emotion prediction.
        """
        if y_true.ndim == 1 and y_pred.ndim == 1:
            # Single emotion dimension
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        else:
            # Multiple emotion dimensions
            distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1))
            return np.mean(distances)

    @classmethod
    def calculate_all_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate all emotion metrics.

        Args:
            y_true: True emotion values
            y_pred: Predicted emotion values
            metric_prefix: Prefix for metric names

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        prefix = f"{metric_prefix}_" if metric_prefix else ""

        # Basic regression metrics
        metrics[f"{prefix}mse"] = cls.mean_squared_error(y_true, y_pred)
        metrics[f"{prefix}rmse"] = cls.root_mean_squared_error(y_true, y_pred)
        metrics[f"{prefix}mae"] = cls.mean_absolute_error(y_true, y_pred)
        metrics[f"{prefix}r2"] = cls.r2_score_metric(y_true, y_pred)
        metrics[f"{prefix}mape"] = cls.mean_absolute_percentage_error(y_true, y_pred)

        # Correlation metrics
        pearson_corr, pearson_p = cls.pearson_correlation(y_true, y_pred)
        spearman_corr, spearman_p = cls.spearman_correlation(y_true, y_pred)

        metrics[f"{prefix}pearson_corr"] = pearson_corr
        metrics[f"{prefix}pearson_p"] = pearson_p
        metrics[f"{prefix}spearman_corr"] = spearman_corr
        metrics[f"{prefix}spearman_p"] = spearman_p

        # Emotion-specific metrics
        metrics[f"{prefix}ccc"] = cls.concordance_correlation_coefficient(y_true, y_pred)
        metrics[f"{prefix}emotion_distance"] = cls.emotion_distance(y_true, y_pred)

        return metrics


class TemporalMetrics:
    """
    Metrics specifically for temporal emotion prediction.
    """

    @staticmethod
    def temporal_consistency(predictions: np.ndarray, window_size: int = 3) -> float:
        """
        Calculate temporal consistency of predictions.
        Measures how smooth the predictions are over time.

        Args:
            predictions: Temporal predictions [seq_len, num_emotions]
            window_size: Window size for consistency calculation

        Returns:
            Temporal consistency score (lower is better)
        """
        if len(predictions) < window_size:
            return 0.0

        consistencies = []
        for i in range(len(predictions) - window_size + 1):
            window = predictions[i:i + window_size]
            # Calculate variance within window
            variance = np.var(window, axis=0)
            consistencies.append(np.mean(variance))

        return np.mean(consistencies)

    @staticmethod
    def trend_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate trend prediction accuracy.
        Measures how well the model captures the direction of emotion changes.

        Args:
            y_true: True emotion sequences [seq_len, num_emotions]
            y_pred: Predicted emotion sequences [seq_len, num_emotions]

        Returns:
            Trend accuracy (0-1, higher is better)
        """
        if len(y_true) < 2:
            return 0.0

        # Calculate differences (trends)
        true_trends = np.diff(y_true, axis=0)
        pred_trends = np.diff(y_pred, axis=0)

        # Calculate sign agreement
        sign_agreement = np.sign(true_trends) == np.sign(pred_trends)

        # Handle zero differences
        zero_mask = (true_trends == 0) & (pred_trends == 0)
        sign_agreement = sign_agreement | zero_mask

        return np.mean(sign_agreement)

    @staticmethod
    def peak_detection_score(y_true: np.ndarray, y_pred: np.ndarray, min_prominence: float = 0.1) -> float:
        """
        Calculate peak detection accuracy.
        Measures how well the model identifies emotional peaks.

        Args:
            y_true: True emotion sequences [seq_len, num_emotions]
            y_pred: Predicted emotion sequences [seq_len, num_emotions]
            min_prominence: Minimum prominence for peak detection

        Returns:
            Peak detection F1 score
        """
        try:
            from scipy.signal import find_peaks
        except ImportError:
            logger.warning("scipy.signal not available for peak detection")
            return 0.0

        if len(y_true) < 3:
            return 0.0

        # Handle multi-dimensional emotions
        if y_true.ndim > 1:
            scores = []
            for dim in range(y_true.shape[1]):
                score = TemporalMetrics.peak_detection_score(
                    y_true[:, dim], y_pred[:, dim], min_prominence
                )
                scores.append(score)
            return np.mean(scores)

        # Find peaks in true and predicted sequences
        true_peaks, _ = find_peaks(y_true, prominence=min_prominence)
        pred_peaks, _ = find_peaks(y_pred, prominence=min_prominence)

        if len(true_peaks) == 0 and len(pred_peaks) == 0:
            return 1.0

        if len(true_peaks) == 0 or len(pred_peaks) == 0:
            return 0.0

        # Calculate overlap of peaks (within tolerance)
        tolerance = max(1, len(y_true) // 20)  # 5% tolerance
        true_positives = 0

        for true_peak in true_peaks:
            if any(abs(pred_peak - true_peak) <= tolerance for pred_peak in pred_peaks):
                true_positives += 1

        precision = true_positives / len(pred_peaks) if len(pred_peaks) > 0 else 0
        recall = true_positives / len(true_peaks) if len(true_peaks) > 0 else 0

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    @staticmethod
    def temporal_mae_weighted(y_true: np.ndarray, y_pred: np.ndarray, decay_factor: float = 0.9) -> float:
        """
        Calculate temporally weighted MAE.
        Recent predictions have higher weights.

        Args:
            y_true: True emotion sequences [seq_len, num_emotions]
            y_pred: Predicted emotion sequences [seq_len, num_emotions]
            decay_factor: Decay factor for temporal weights

        Returns:
            Weighted MAE
        """
        seq_len = len(y_true)
        weights = np.array([decay_factor ** (seq_len - i - 1) for i in range(seq_len)])
        weights = weights / weights.sum()  # Normalize weights

        absolute_errors = np.abs(y_true - y_pred)
        if absolute_errors.ndim > 1:
            absolute_errors = np.mean(absolute_errors, axis=1)

        weighted_mae = np.sum(weights * absolute_errors)
        return weighted_mae

    @classmethod
    def calculate_temporal_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate all temporal metrics.

        Args:
            y_true: True emotion sequences
            y_pred: Predicted emotion sequences
            metric_prefix: Prefix for metric names

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        prefix = f"{metric_prefix}_" if metric_prefix else ""

        # Temporal-specific metrics
        metrics[f"{prefix}temporal_consistency"] = cls.temporal_consistency(y_pred)
        metrics[f"{prefix}trend_accuracy"] = cls.trend_accuracy(y_true, y_pred)
        metrics[f"{prefix}peak_detection"] = cls.peak_detection_score(y_true, y_pred)
        metrics[f"{prefix}temporal_mae_weighted"] = cls.temporal_mae_weighted(y_true, y_pred)

        return metrics


class MetricsCalculator:
    """
    Comprehensive metrics calculator for emotion prediction.
    """

    def __init__(self, metrics_config: Optional[Dict] = None):
        """
        Initialize the metrics calculator.

        Args:
            metrics_config: Configuration for metrics calculation
        """
        self.config = metrics_config or {}
        self.emotion_metrics = EmotionMetrics()
        self.temporal_metrics = TemporalMetrics()

    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        temporal: bool = False,
        per_dimension: bool = True,
        user_ids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.

        Args:
            y_true: True emotion values
            y_pred: Predicted emotion values
            temporal: Whether to calculate temporal metrics
            per_dimension: Whether to calculate metrics per emotion dimension
            user_ids: Optional user IDs for per-user metrics

        Returns:
            Dictionary of calculated metrics
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        metrics = {}

        # Overall metrics
        overall_metrics = self.emotion_metrics.calculate_all_metrics(
            y_true.flatten(), y_pred.flatten(), "overall"
        )
        metrics.update(overall_metrics)

        # Per-dimension metrics
        if per_dimension and y_true.ndim > 1 and y_true.shape[-1] > 1:
            dimension_names = ["valence", "arousal"] if y_true.shape[-1] == 2 else [f"dim_{i}" for i in range(y_true.shape[-1])]

            for i, dim_name in enumerate(dimension_names):
                if i < y_true.shape[-1]:
                    dim_metrics = self.emotion_metrics.calculate_all_metrics(
                        y_true[..., i], y_pred[..., i], dim_name
                    )
                    metrics.update(dim_metrics)

        # Temporal metrics
        if temporal and y_true.ndim >= 2:
            if y_true.ndim == 3:  # [batch_size, seq_len, num_emotions]
                # Average temporal metrics across batch
                temp_metrics_list = []
                for i in range(y_true.shape[0]):
                    temp_metrics = self.temporal_metrics.calculate_temporal_metrics(
                        y_true[i], y_pred[i], "temporal"
                    )
                    temp_metrics_list.append(temp_metrics)

                # Average across batch
                for key in temp_metrics_list[0].keys():
                    metrics[key] = np.mean([tm[key] for tm in temp_metrics_list])
            else:  # [seq_len, num_emotions]
                temp_metrics = self.temporal_metrics.calculate_temporal_metrics(
                    y_true, y_pred, "temporal"
                )
                metrics.update(temp_metrics)

        # Per-user metrics
        if user_ids is not None:
            user_metrics = self._calculate_per_user_metrics(y_true, y_pred, user_ids)
            metrics.update(user_metrics)

        return metrics

    def _calculate_per_user_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        user_ids: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics per user and aggregate."""
        user_metrics = {}
        unique_users = np.unique(user_ids)

        user_scores = {"mse": [], "mae": [], "pearson_corr": [], "ccc": []}

        for user_id in unique_users:
            user_mask = user_ids == user_id
            user_true = y_true[user_mask]
            user_pred = y_pred[user_mask]

            if len(user_true) > 1:  # Ensure sufficient data
                user_true_flat = user_true.flatten()
                user_pred_flat = user_pred.flatten()

                user_scores["mse"].append(self.emotion_metrics.mean_squared_error(user_true_flat, user_pred_flat))
                user_scores["mae"].append(self.emotion_metrics.mean_absolute_error(user_true_flat, user_pred_flat))

                pearson_corr, _ = self.emotion_metrics.pearson_correlation(user_true_flat, user_pred_flat)
                user_scores["pearson_corr"].append(pearson_corr)

                ccc = self.emotion_metrics.concordance_correlation_coefficient(user_true_flat, user_pred_flat)
                user_scores["ccc"].append(ccc)

        # Aggregate per-user metrics
        for metric_name, scores in user_scores.items():
            if scores:
                user_metrics[f"user_avg_{metric_name}"] = np.mean(scores)
                user_metrics[f"user_std_{metric_name}"] = np.std(scores)

        return user_metrics

    def format_metrics(self, metrics: Dict[str, float], round_digits: int = 4) -> Dict[str, float]:
        """Format metrics for display."""
        return {key: round(value, round_digits) for key, value in metrics.items()}

    def save_metrics(self, metrics: Dict[str, float], filepath: str):
        """Save metrics to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

    def load_metrics(self, filepath: str) -> Dict[str, float]:
        """Load metrics from file."""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)