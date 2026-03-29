from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class IrisDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y)

    def get_binary_subset(self, class0: str, class1: str) -> "IrisDataset":
        target_map = {
            self.target_names.index(class0): 0,
            self.target_names.index(class1): 1,
        }
        mask = np.isin(self.y, list(target_map.keys()))
        X_binary = self.X[mask]
        y_binary = np.array([target_map[y] for y in self.y[mask]])

        train_mask = np.isin(self.y_train, list(target_map.keys()))
        test_mask = np.isin(self.y_test, list(target_map.keys()))

        return IrisDataset(
            X=X_binary,
            y=y_binary,
            feature_names=self.feature_names,
            target_names=[class0, class1],
            X_train=self.X_train[train_mask],
            X_test=self.X_test[test_mask],
            y_train=self.y_train[train_mask],
            y_test=self.y_test[test_mask],
            scaler=self.scaler,
        )


def load_iris_data(
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> IrisDataset:
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)

    return IrisDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )


def generate_synthetic_datasets(
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from sklearn.datasets import make_circles, make_classification, make_moons

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    X_linear, y_linear = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )
    datasets["linearly_separable"] = (X_linear, y_linear)

    X_circles, y_circles = make_circles(
        n_samples=200,
        noise=0.05,
        factor=0.5,
        random_state=random_state,
    )
    datasets["concentric_circles"] = (X_circles, y_circles)

    X_moons, y_moons = make_moons(
        n_samples=200,
        noise=0.1,
        random_state=random_state,
    )
    datasets["moon_shapes"] = (X_moons, y_moons)

    return datasets
