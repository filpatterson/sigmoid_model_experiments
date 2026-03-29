import numpy as np
import pytest
from sklearn.datasets import make_classification

from svm_demo import (
    SupportVectorMachine,
    generate_synthetic_datasets,
    load_iris_data,
    train_binary_model,
    train_model,
)


@pytest.fixture
def iris_data():
    return load_iris_data(test_size=0.2, random_state=42)


@pytest.fixture
def synthetic_data():
    return generate_synthetic_datasets(random_state=42)


@pytest.fixture
def simple_binary_data():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42,
    )
    return X, y


class TestSupportVectorMachine:
    def test_svm_creation_default(self):
        svm = SupportVectorMachine()
        assert svm.kernel == "rbf"
        assert svm.C == 1.0
        assert svm.gamma == "scale"

    def test_svm_creation_custom(self):
        svm = SupportVectorMachine(kernel="linear", C=10.0, gamma=0.01)
        assert svm.kernel == "linear"
        assert svm.C == 10.0
        assert svm.gamma == 0.01

    def test_svm_fit_predict(self, simple_binary_data):
        X, y = simple_binary_data
        svm = SupportVectorMachine(kernel="linear", C=1.0, random_state=42)
        svm.fit(X, y)
        predictions = svm.predict(X)
        assert predictions.shape == y.shape
        assert np.all(np.isin(predictions, [0, 1]))

    def test_svm_score(self, simple_binary_data):
        X, y = simple_binary_data
        svm = SupportVectorMachine(kernel="linear", random_state=42)
        svm.fit(X, y)
        score = svm.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_svm_support_vectors(self, simple_binary_data):
        X, y = simple_binary_data
        svm = SupportVectorMachine(kernel="linear", random_state=42)
        svm.fit(X, y)
        assert svm.support_vectors_.shape[0] > 0
        assert svm.support_vectors_.shape[1] == 2

    def test_svm_rbf_kernel(self, simple_binary_data):
        X, y = simple_binary_data
        svm = SupportVectorMachine(kernel="rbf", gamma="scale", random_state=42)
        svm.fit(X, y)
        assert svm.predict(X).shape == y.shape

    def test_svm_poly_kernel(self, simple_binary_data):
        X, y = simple_binary_data
        svm = SupportVectorMachine(kernel="poly", degree=3, random_state=42)
        svm.fit(X, y)
        assert svm.predict(X).shape == y.shape

    def test_svm_empty_data_raises(self):
        svm = SupportVectorMachine()
        with pytest.raises(ValueError):
            svm.fit(np.array([]).reshape(0, 2), np.array([]))

    def test_svm_unfitted_predict_raises(self):
        svm = SupportVectorMachine()
        with pytest.raises(RuntimeError):
            svm.predict(np.array([[1, 2]]))


class TestLoadIrisData:
    def test_load_iris_returns_correct_shape(self, iris_data):
        assert iris_data.X.shape == (150, 4)
        assert iris_data.y.shape == (150,)

    def test_load_iris_has_3_classes(self, iris_data):
        assert len(iris_data.classes) == 3

    def test_load_iris_train_test_split(self, iris_data):
        assert iris_data.X_train.shape[0] + iris_data.X_test.shape[0] == 150

    def test_load_iris_feature_names(self, iris_data):
        assert len(iris_data.feature_names) == 4

    def test_load_iris_target_names(self, iris_data):
        assert iris_data.target_names == ["setosa", "versicolor", "virginica"]


class TestGenerateSyntheticData:
    def test_generate_circles(self, synthetic_data):
        X, y = synthetic_data["concentric_circles"]
        assert X.shape[0] == 200
        assert X.shape[1] == 2
        assert len(np.unique(y)) == 2

    def test_generate_moons(self, synthetic_data):
        X, y = synthetic_data["moon_shapes"]
        assert X.shape[0] == 200
        assert X.shape[1] == 2
        assert len(np.unique(y)) == 2

    def test_generate_linear(self, synthetic_data):
        X, y = synthetic_data["linearly_separable"]
        assert X.shape[0] == 200
        assert X.shape[1] == 2
        assert len(np.unique(y)) == 2


class TestTrainModel:
    def test_train_model_returns_dict(self, iris_data):
        result = train_model(iris_data, kernel="linear")
        assert isinstance(result, dict)
        assert "model" in result
        assert "accuracy" in result

    def test_train_model_accuracy(self, iris_data):
        result = train_model(iris_data, kernel="linear")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_train_binary_model(self, simple_binary_data):
        X, y = simple_binary_data
        result = train_binary_model(X, y, kernel="linear")
        assert result["accuracy"] >= 0.0


class TestIrisBinarySubset:
    def test_get_binary_subset(self, iris_data):
        subset = iris_data.get_binary_subset("setosa", "versicolor")
        assert len(np.unique(subset.y)) == 2
        assert subset.target_names == ["setosa", "versicolor"]
