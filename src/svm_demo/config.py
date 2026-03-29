RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

GRID_SEARCH_PARAMS: dict[str, list[float | str]] = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
}

BINARY_CLASSES: tuple[str, str] = ("setosa", "versicolor")
IRIS_CLASSES: tuple[str, str, str] = ("setosa", "versicolor", "virginica")

FEATURE_NAMES: list[str] = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
