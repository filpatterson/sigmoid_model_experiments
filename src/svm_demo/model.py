from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC


@dataclass
class SupportVectorMachine:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: float | str = "scale"
    degree: int = 3
    coef0: float = 0.0
    probability: bool = False
    random_state: int | None = None

    _model: SVC | None = None

    def __post_init__(self) -> None:
        self._is_fitted = False
        self._model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=self.probability,
            random_state=self.random_state,
        )

    @property
    def model(self) -> SVC:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._model

    @property
    def support_vectors_(self) -> np.ndarray:
        return self.model.support_vectors_

    @property
    def n_support_vectors(self) -> np.ndarray:
        return self.model.n_support_

    @property
    def support_dual_coef_(self) -> np.ndarray:
        return self.model.dual_coef_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorMachine":
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y must not be empty.")
        self._model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=self.probability,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)

    def get_params(self) -> dict[str, float | str | int]:
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
        }
