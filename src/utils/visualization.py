"""
Visualization utilities for SemEval-2026 Task 2 project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot training curves for loss and metrics.

    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        train_metrics: Training metrics over epochs
        val_metrics: Validation metrics over epochs
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Determine number of subplots
    n_metrics = len(train_metrics) if train_metrics else 0
    n_plots = 1 + n_metrics  # 1 for loss + n for metrics

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metrics
    if train_metrics:
        for i, (metric_name, values) in enumerate(train_metrics.items()):
            ax = axes[i + 1]
            ax.plot(epochs, values, 'b-', label=f'Training {metric_name}', linewidth=2)

            if val_metrics and metric_name in val_metrics:
                ax.plot(epochs, val_metrics[metric_name], 'r-',
                       label=f'Validation {metric_name}', linewidth=2)

            ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")

    plt.show()


def plot_emotion_distribution(
    data: pd.DataFrame,
    valence_col: str = 'valence',
    arousal_col: str = 'arousal',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot emotion distribution (valence and arousal).

    Args:
        data: DataFrame with emotion data
        valence_col: Column name for valence
        arousal_col: Column name for arousal
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot valence distribution
    axes[0].hist(data[valence_col], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Valence Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Valence')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Plot arousal distribution
    axes[1].hist(data[arousal_col], bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title('Arousal Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Arousal')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    # Plot valence vs arousal
    scatter = axes[2].scatter(data[valence_col], data[arousal_col], alpha=0.6, s=20)
    axes[2].set_title('Valence vs Arousal', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Valence')
    axes[2].set_ylabel('Arousal')
    axes[2].grid(True, alpha=0.3)

    # Add quadrant lines
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Emotion distribution plot saved to {save_path}")

    plt.show()


def plot_temporal_patterns(
    data: pd.DataFrame,
    user_col: str = 'user_id',
    time_col: str = 'timestamp',
    valence_col: str = 'valence',
    arousal_col: str = 'arousal',
    n_users: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot temporal patterns for selected users.

    Args:
        data: DataFrame with temporal emotion data
        user_col: Column name for user ID
        time_col: Column name for timestamp
        valence_col: Column name for valence
        arousal_col: Column name for arousal
        n_users: Number of users to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert timestamp to datetime if it's not already
    if data[time_col].dtype == 'object':
        data[time_col] = pd.to_datetime(data[time_col])

    # Select users with most data points
    user_counts = data[user_col].value_counts()
    top_users = user_counts.head(n_users).index

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    colors = plt.cm.Set3(np.linspace(0, 1, n_users))

    for i, user_id in enumerate(top_users):
        user_data = data[data[user_col] == user_id].sort_values(time_col)

        # Plot valence over time
        axes[0].plot(user_data[time_col], user_data[valence_col],
                    marker='o', label=f'User {user_id}', color=colors[i],
                    linewidth=2, markersize=4)

        # Plot arousal over time
        axes[1].plot(user_data[time_col], user_data[arousal_col],
                    marker='s', label=f'User {user_id}', color=colors[i],
                    linewidth=2, markersize=4)

    axes[0].set_title('Valence Temporal Patterns', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Valence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Arousal Temporal Patterns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Arousal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Temporal patterns plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Path to save the plot
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_correlation_heatmap(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot correlation heatmap.

    Args:
        data: DataFrame with numerical data
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=[np.number])

    plt.figure(figsize=figsize)
    correlation_matrix = numerical_data.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {save_path}")

    plt.show()


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> None:
    """
    Plot prediction scatter plot.

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate and display R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction scatter plot saved to {save_path}")

    plt.show()


def plot_learning_rate_schedule(
    learning_rates: List[float],
    steps: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot learning rate schedule.

    Args:
        learning_rates: Learning rates over training
        steps: Training steps (if None, uses indices)
        save_path: Path to save the plot
        figsize: Figure size
    """
    if steps is None:
        steps = list(range(len(learning_rates)))

    plt.figure(figsize=figsize)
    plt.plot(steps, learning_rates, linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning rate schedule plot saved to {save_path}")

    plt.show()


def plot_error_analysis(
    errors: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot error analysis.

    Args:
        errors: Prediction errors
        predictions: Predicted values
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Error distribution
    axes[0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Error vs predictions
    axes[1].scatter(predictions, errors, alpha=0.6, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1].set_title('Error vs Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predictions')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)

    # Error magnitude vs predictions
    axes[2].scatter(predictions, np.abs(errors), alpha=0.6, s=20)
    axes[2].set_title('Absolute Error vs Predictions', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Predictions')
    axes[2].set_ylabel('Absolute Error')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Error analysis plot saved to {save_path}")

    plt.show()


def save_all_plots(plots_dir: str = "plots") -> None:
    """
    Save all open plots to directory.

    Args:
        plots_dir: Directory to save plots
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get all figure numbers
    fig_nums = plt.get_fignums()

    for fig_num in fig_nums:
        fig = plt.figure(fig_num)
        fig_name = f"figure_{fig_num}.png"
        fig_path = plots_dir / fig_name
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure {fig_num} to {fig_path}")


def close_all_plots() -> None:
    """Close all open plots."""
    plt.close('all')
    logger.debug("Closed all plots")