"""Data loading and preprocessing for Random Forest demo."""

from typing import Any

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
