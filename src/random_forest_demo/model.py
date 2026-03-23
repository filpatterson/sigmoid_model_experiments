"""Random Forest classifier implementation."""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRFClassifier


class RandomForestClassifier:
    """Random Forest classifier wrapper.

    A wrapper around scikit-learn's RandomForestClassifier with additional
    convenience methods and type annotations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees. None means unlimited.
            min_samples_split: Minimum samples required to split an internal node.
            min_samples_leaf: Minimum samples required at a leaf node.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model = SklearnRFClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "RandomForestClassifier":
        """Fit the classifier to training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Samples of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Samples of shape (n_samples, n_features).

        Returns:
            Probability estimates of shape (n_samples, n_classes).
        """
        return self._model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on given data.

        Args:
            X: Features of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            Accuracy score.
        """
        return self._model.score(X, y)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances from the forest."""
        return self._model.feature_importances_

    @property
    def classes_(self) -> np.ndarray:
        """Get unique class labels."""
        return self._model.classes_

    @property
    def n_classes_(self) -> int:
        """Get number of classes."""
        return len(self._model.classes_)
