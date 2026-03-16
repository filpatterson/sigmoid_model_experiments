"""Naive Bayes Demo - Breast Cancer Classification.

A demonstration of Gaussian Naive Bayes classifier performance on the
Breast Cancer Wisconsin dataset, showcasing strong results with a
simple probabilistic model.
"""

from naive_bayes_demo.data import load_breast_cancer
from naive_bayes_demo.model import NaiveBayesClassifier
from naive_bayes_demo.train import train_model
from naive_bayes_demo.evaluate import evaluate_model, plot_confusion_matrix

__all__ = [
    "load_breast_cancer",
    "NaiveBayesClassifier",
    "train_model",
    "evaluate_model",
    "plot_confusion_matrix",
]
