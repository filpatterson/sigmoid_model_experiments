from svm_demo.data import IrisDataset, generate_synthetic_datasets, load_iris_data
from svm_demo.evaluate import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_grid_search_heatmap,
    plot_multiple_decision_boundaries,
    plot_svm_margin_diagram,
    print_classification_metrics,
)
from svm_demo.model import SupportVectorMachine
from svm_demo.train import (
    ModelTrainingError,
    grid_search_svm,
    train_binary_model,
    train_model,
)

__all__ = [
    "SupportVectorMachine",
    "ModelTrainingError",
    "load_iris_data",
    "generate_synthetic_datasets",
    "train_model",
    "train_binary_model",
    "grid_search_svm",
    "IrisDataset",
    "plot_decision_boundary",
    "plot_confusion_matrix",
    "plot_grid_search_heatmap",
    "plot_multiple_decision_boundaries",
    "plot_svm_margin_diagram",
    "print_classification_metrics",
]
