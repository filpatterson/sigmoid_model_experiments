"""Unit tests for Random Forest demo."""

import numpy as np

from random_forest_demo.data import load_breast_cancer_data
from random_forest_demo.model import RandomForestClassifier
from random_forest_demo.train import train_model
from random_forest_demo.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_breast_cancer_data_returns_dict(self):
        """Test that load_breast_cancer_data returns expected keys."""
        result = load_breast_cancer_data()
        expected_keys = {
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "feature_names",
            "target_names",
        }
        assert expected_keys.issubset(result.keys())

    def test_train_test_split_proportions(self):
        """Test that train/test split is 80/20 by default."""
        result = load_breast_cancer_data(test_size=0.2)
        n_train = len(result["X_train"])
        n_test = len(result["X_test"])
        total = n_train + n_test
        assert abs(n_test / total - 0.2) < 0.01

    def test_train_test_split_stratified(self):
        """Test that stratified split maintains class proportions."""
        result = load_breast_cancer_data(stratify=True)
        train_class_ratio = np.mean(result["y_train"])
        test_class_ratio = np.mean(result["y_test"])
        assert abs(train_class_ratio - test_class_ratio) < 0.05

    def test_train_test_split_reproducible(self):
        """Test that same random_state produces same splits."""
        result1 = load_breast_cancer_data(random_state=42)
        result2 = load_breast_cancer_data(random_state=42)
        assert np.array_equal(result1["X_train"], result2["X_train"])
        assert np.array_equal(result1["y_train"], result2["y_train"])


class TestRandomForestClassifier:
    """Tests for Random Forest classifier."""

    def test_model_fit_returns_self(self):
        """Test that fit method returns self for chaining."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = model.fit(X, y)
        assert result is model

    def test_model_predict_returns_correct_shape(self):
        """Test that predict returns correct number of predictions."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (4,)

    def test_model_predict_proba_sum_to_one(self):
        """Test that predicted probabilities sum to 1."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_model_score_returns_accuracy(self):
        """Test that score returns accuracy between 0 and 1."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_model_feature_importances(self):
        """Test that feature importances are returned."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        assert importances.shape == (3,)
        assert np.all(importances >= 0)
        assert np.isclose(importances.sum(), 1.0, atol=1e-10)


class TestTrainModel:
    """Tests for training pipeline."""

    def test_train_model_returns_all_keys(self):
        """Test that train_model returns all expected outputs."""
        result = train_model()
        expected_keys = {
            "model",
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "y_pred",
            "y_proba",
            "accuracy",
            "feature_importances",
        }
        assert expected_keys.issubset(result.keys())

    def test_train_model_accuracy_is_valid(self):
        """Test that accuracy is a valid probability."""
        result = train_model()
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_train_model_predictions_match_test_size(self):
        """Test that predictions match test set size."""
        result = train_model()
        assert len(result["y_pred"]) == len(result["y_test"])

    def test_train_model_feature_importances_shape(self):
        """Test that feature importances match feature count."""
        result = train_model()
        data = load_breast_cancer_data()
        n_features = data["X_train"].shape[1]
        assert result["feature_importances"].shape == (n_features,)


class TestEvaluateModel:
    """Tests for evaluation functions."""

    def test_evaluate_model_returns_metrics(self):
        """Test that evaluate_model returns expected metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])

        metrics = evaluate_model(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics

    def test_evaluate_model_perfect_accuracy(self):
        """Test that perfect predictions give 1.0 accuracy."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = evaluate_model(y_true, y_pred)

        assert metrics["accuracy"] == 1.0

    def test_evaluate_model_zero_accuracy(self):
        """Test that all wrong predictions give 0.0 accuracy."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        metrics = evaluate_model(y_true, y_pred)

        assert metrics["accuracy"] == 0.0


class TestPlotting:
    """Tests for visualization functions."""

    def test_plot_confusion_matrix_returns_figure(self):
        """Test that plot_confusion_matrix returns a figure."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None

    def test_plot_feature_importance_returns_figure(self):
        """Test that plot_feature_importance returns a figure."""
        feature_importances = np.array([0.2, 0.5, 0.3])
        feature_names = np.array(["a", "b", "c"])
        fig = plot_feature_importance(feature_importances, feature_names)
        assert fig is not None

    def test_plot_roc_curve_returns_figure(self):
        """Test that plot_roc_curve returns a figure."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])
        fig = plot_roc_curve(y_true, y_proba)
        assert fig is not None
