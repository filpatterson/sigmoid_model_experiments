from typing import Any

import numpy as np

from svm_demo.config import GRID_SEARCH_PARAMS, RANDOM_STATE, TEST_SIZE
from svm_demo.data import IrisDataset
from svm_demo.model import SupportVectorMachine


class ModelTrainingError(Exception):
    pass


def train_model(
    dataset: IrisDataset,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float | str = "scale",
    probability: bool = False,
) -> dict[str, Any]:
    if len(dataset.X_train) == 0:
        raise ModelTrainingError("Training data is empty.")
    if len(dataset.y_train) == 0:
        raise ModelTrainingError("Training labels are empty.")

    try:
        svm = SupportVectorMachine(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=RANDOM_STATE,
        )
        svm.fit(dataset.X_train, dataset.y_train)
    except Exception as e:
        raise ModelTrainingError(f"Failed to train SVM model: {e}") from e

    y_pred = svm.predict(dataset.X_test)
    accuracy = svm.score(dataset.X_test, dataset.y_test)

    result: dict[str, Any] = {
        "model": svm,
        "X_train": dataset.X_train,
        "X_test": dataset.X_test,
        "y_train": dataset.y_train,
        "y_test": dataset.y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
    }

    if probability:
        try:
            y_proba = svm.model.predict_proba(dataset.X_test)
            result["y_proba"] = y_proba
        except Exception:
            pass

    return result


def train_binary_model(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float | str = "scale",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    svm = SupportVectorMachine(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=random_state,
    )
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)

    return {
        "model": svm,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
    }


def grid_search_svm(
    dataset: IrisDataset,
    param_grid: dict[str, list[float | str]] | None = None,
    cv: int = 5,
) -> dict[str, Any]:
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    if param_grid is None:
        param_grid = GRID_SEARCH_PARAMS

    base_model = SVC(random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )

    grid_search.fit(dataset.X_train, dataset.y_train)

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_,
        "grid_search": grid_search,
    }
