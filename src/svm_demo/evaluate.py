from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Decision Boundary",
    kernel: str = "rbf",
    resolution: float = 0.02,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        if hasattr(Z, "reshape"):
            Z = Z.reshape(xx.shape)
    except Exception:
        Z = np.zeros(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.5)

    unique_labels = np.unique(y)
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
    for idx, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[idx % len(colors)],
            label=f"Class {label}",
            edgecolors="black",
            s=50,
            alpha=0.8,
        )

    if hasattr(model, "support_vectors_"):
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            label="Support Vectors",
        )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"{title} (Kernel: {kernel})")
    ax.legend()

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Confusion Matrix",
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title(title)

    return ax


def plot_grid_search_heatmap(
    cv_results: dict[str, Any],
    param_c: str = "param_C",
    param_gamma: str = "param_gamma",
    ax: plt.Axes | None = None,
    title: str = "Grid Search Results (C vs gamma)",
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    mask = cv_results["param_kernel"] == "rbf"
    c_values = cv_results[param_c][mask].astype(float)
    gamma_values = cv_results[param_gamma][mask]
    scores = cv_results["mean_test_score"][mask]

    c_unique = np.sort(np.unique(c_values))
    gamma_unique = np.sort(np.unique(gamma_values))

    score_grid = np.zeros((len(gamma_unique), len(c_unique)))

    for c_idx, c_val in enumerate(c_unique):
        for gamma_idx, gamma_val in enumerate(gamma_unique):
            for i in range(len(scores)):
                if c_values[i] == c_val and gamma_values[i] == gamma_val:
                    score_grid[gamma_idx, c_idx] = scores[i]

    im = ax.imshow(
        score_grid,
        cmap="viridis",
        aspect="auto",
        origin="lower",
    )
    ax.set_xticks(np.arange(len(c_unique)))
    ax.set_yticks(np.arange(len(gamma_unique)))
    ax.set_xticklabels([f"{c:.2f}" for c in c_unique])
    ax.set_yticklabels([str(g) for g in gamma_unique])
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Accuracy")

    for i in range(len(gamma_unique)):
        for j in range(len(c_unique)):
            ax.text(
                j,
                i,
                f"{score_grid[i, j]:.3f}",
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )

    return ax


def plot_multiple_decision_boundaries(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    titles: dict[str, str] | None = None,
    resolution: float = 0.02,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(models.items()):
        if idx >= 4:
            break
        title = titles.get(name, f"Kernel: {name}") if titles else f"Kernel: {name}"
        plot_decision_boundary(
            model=model,
            X=X,
            y=y,
            ax=axes[idx],
            title=title,
            kernel=name,
            resolution=resolution,
        )

    for idx in range(len(models), 4):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> None:
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print("=" * 60)
    print(report)


def plot_svm_margin_diagram(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    class0_mask = y == 0
    class1_mask = y == 1

    ax.scatter(
        X[class0_mask, 0],
        X[class0_mask, 1],
        c="red",
        s=80,
        label="Class 0",
        edgecolors="k",
        zorder=5,
    )
    ax.scatter(
        X[class1_mask, 0],
        X[class1_mask, 1],
        c="blue",
        s=80,
        label="Class 1",
        edgecolors="k",
        zorder=5,
    )

    if hasattr(model, "support_vectors_"):
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=250,
            facecolors="none",
            edgecolors="gold",
            linewidths=3,
            label="Support Vectors",
            zorder=10,
        )

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    if hasattr(model, "model") and hasattr(model.model, "coef_"):
        w = model.model.coef_[0]
        b = model.model.intercept_[0]
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, "k-", linewidth=2, label="Decision Boundary")

        margin = 2 / np.linalg.norm(w)
        y_parallel1 = y_line + margin / 2
        y_parallel2 = y_line - margin / 2
        ax.plot(
            x_line, y_parallel1, "g--", linewidth=1.5, alpha=0.7, label="Margin Bounds"
        )
        ax.plot(x_line, y_parallel2, "g--", linewidth=1.5, alpha=0.7)

        ax.annotate(
            "Maximum Margin",
            xy=(x_line[len(x_line) // 2], y_parallel1[len(x_line) // 2]),
            xytext=(x_line[len(x_line) // 2] + 1, y_parallel1[len(x_line) // 2] + 1),
            arrowprops=dict(arrowstyle="<->", color="green"),
            fontsize=12,
            color="green",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title("SVM Maximum Margin Classifier", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax
