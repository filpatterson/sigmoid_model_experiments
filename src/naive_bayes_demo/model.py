"""Naive Bayes classifier implementation."""

from typing import Optional

import numpy as np
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier:
    """Gaussian Naive Bayes classifier wrapper.

    A wrapper around scikit-learn's GaussianNB with additional
    convenience methods and type annotations.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize the classifier.

        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter.
        """
        self.alpha = alpha
        self._model = GaussianNB(alpha=alpha)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "NaiveBayesClassifier":
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
    def class_prior_(self) -> Optional[np.ndarray]:
        """Get class prior probabilities."""
        return self._model.class_prior_

    @property
    def theta_(self) -> Optional[np.ndarray]:
        """Get mean of each feature per class."""
        return self._model.theta_

    @property
    def var_(self) -> Optional[np.ndarray]:
        """Get variance of each feature per class."""
        return self._model.var_

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Get unique class labels."""
        return self._model.classes_
