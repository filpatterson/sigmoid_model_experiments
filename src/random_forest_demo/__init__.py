"""Random Forest Demo - Breast Cancer Classification.

A demonstration of Random Forest classifier performance on the
Breast Cancer Wisconsin dataset, showcasing ensemble learning with
decision trees.
"""

from random_forest_demo.data import load_breast_cancer_data
from random_forest_demo.model import RandomForestClassifier
from random_forest_demo.train import train_model
from random_forest_demo.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)

__all__ = [
    "load_breast_cancer_data",
    "RandomForestClassifier",
    "train_model",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
]
