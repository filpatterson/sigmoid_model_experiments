"""Training pipeline for Random Forest classifier."""

from typing import Any

from random_forest_demo.config import (
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_ESTIMATORS,
    RANDOM_STATE,
    STRATIFY,
    TEST_SIZE,
)
from random_forest_demo.data import load_breast_cancer_data
from random_forest_demo.model import RandomForestClassifier


class ModelTrainingError(Exception):
    """Raised when model training fails."""

    pass


def train_model(
    n_estimators: int = N_ESTIMATORS,
    max_depth: int | None = MAX_DEPTH,
    min_samples_split: int = MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    stratify: bool = STRATIFY,
) -> dict[str, Any]:
    """Train a Random Forest classifier on breast cancer data.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees. None means unlimited.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        random_state: Random seed for reproducibility.
        test_size: Proportion of data to use for testing.
        stratify: Whether to stratify splits by class label.

    Returns:
        Dictionary containing:
            - model: Trained RandomForestClassifier
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - y_pred: Predicted labels on test set
            - y_proba: Predicted probabilities on test set
            - accuracy: Test accuracy score
            - feature_importances: Feature importance scores

    Raises:
        ModelTrainingError: If training data is invalid.
    """
    data = load_breast_cancer_data(
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

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
            f"X_test and y_test must have same length: {len(X_test)} != {len(y_test)}"
        )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)
    feature_importances = model.feature_importances_

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "feature_importances": feature_importances,
    }
