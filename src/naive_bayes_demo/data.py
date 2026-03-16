"""Data loading and preprocessing for Naive Bayes demo."""

from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict[str, Any]:
    """Load and split the Breast Cancer Wisconsin dataset.

    Args:
        test_size: Proportion of data to use for testing.
        random_state: Seed for random number generator.
        stratify: Whether to stratify splits by class label.

    Returns:
        Dictionary containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - feature_names: Names of features
            - target_names: Names of target classes
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_names": target_names,
    }


def standardize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using z-score normalization.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, mean, std).
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


def get_feature_statistics(
    X: np.ndarray,
    y: np.ndarray,
    class_idx: int,
) -> dict[str, np.ndarray]:
    """Calculate mean and std for each feature within a class.

    Args:
        X: Feature matrix.
        y: Class labels.
        class_idx: Index of the class to compute statistics for.

    Returns:
        Dictionary with 'mean' and 'std' arrays.
    """
    X_class = X[y == class_idx]
    return {
        "mean": np.mean(X_class, axis=0),
        "std": np.std(X_class, axis=0),
    }
