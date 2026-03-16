"""Training pipeline for Naive Bayes classifier."""

from typing import Any

import numpy as np

from naive_bayes_demo.data import load_breast_cancer_data, standardize_features
from naive_bayes_demo.model import NaiveBayesClassifier


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


def train_model(
    use_standardization: bool = False,
    alpha: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a Gaussian Naive Bayes classifier on breast cancer data.

    Args:
        use_standardization: Whether to standardize features before training.
        alpha: Smoothing parameter for GaussianNB.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - model: Trained NaiveBayesClassifier
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - y_pred: Predicted labels on test set
            - y_proba: Predicted probabilities on test set
            - accuracy: Test accuracy score

    Raises:
        ModelTrainingError: If training data is invalid.
    """
    data = load_breast_cancer_data(random_state=random_state)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    if len(X_train) == 0 or len(y_train) == 0:
        raise ModelTrainingError("Training data must not be empty")
    if len(X_train) != len(y_train):
        raise ModelTrainingError(
            f"X_train and y_train must have same length: "
            f"{len(X_train)} != {len(y_train)}"
        )
    if len(X_test) != len(y_test):
        raise ModelTrainingError(
            f"X_test and y_test must have same length: "
            f"{len(X_test)} != {len(y_test)}"
        )

    if use_standardization:
        X_train, X_test, _, _ = standardize_features(X_train, X_test)

    model = NaiveBayesClassifier(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
    }
