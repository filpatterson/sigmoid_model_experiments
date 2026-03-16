"""Model evaluation and visualization."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    target_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Evaluate model performance with multiple metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_proba: Predicted class probabilities (optional, for AUC).
        target_names: Names of target classes.

    Returns:
        Dictionary containing evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        if y_proba.shape[1] == 2:
            metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")

    if target_names is not None:
        metrics["report"] = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
    else:
        metrics["report"] = classification_report(y_true, y_pred, zero_division=0)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        target_names: Names of target classes.
        title: Plot title.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names if target_names else "auto",
        yticklabels=target_names if target_names else "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot ROC curve.

    Args:
        y_true: True class labels.
        y_proba: Predicted class probabilities.
        title: Plot title.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    if y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_proba[:, 1])
    else:
        fpr, tpr, _ = roc_curve(y_true, y_proba.ravel())
        auc_score = roc_auc_score(y_true, y_proba, multi_class="ovr")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: np.ndarray,
    title: str = "Feature Importance (Mean Difference Between Classes)",
    top_n: Optional[int] = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot feature importance based on mean difference between classes.

    Args:
        X: Feature matrix.
        y: Class labels.
        feature_names: Names of features.
        title: Plot title.
        top_n: Number of top features to show (None for all).
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    class_0_mean = np.mean(X[y == 0], axis=0)
    class_1_mean = np.mean(X[y == 1], axis=0)
    importance = np.abs(class_1_mean - class_0_mean)

    if top_n is not None:
        top_indices = np.argsort(importance)[-top_n:]
        importance = importance[top_indices]
        feature_names = feature_names[top_indices]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Absolute Mean Difference")
    ax.set_title(title)
    ax.invert_yaxis()

    return fig


def plot_class_distribution(
    y: np.ndarray,
    target_names: list[str],
    title: str = "Class Distribution",
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:
    """Plot distribution of classes.

    Args:
        y: Class labels.
        target_names: Names of target classes.
        title: Plot title.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([target_names[i] for i in unique], counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)

    for i, count in enumerate(counts):
        ax.text(i, count + 5, str(count), ha="center")

    return fig
