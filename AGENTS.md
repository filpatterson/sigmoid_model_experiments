# AGENTS.md - Sigmoid Model Experiments

## Project Overview

This repository contains classification and regression model implementations using industry-standard ML libraries.
Each project is organized in its own directory with the following structure:
- `src/{project_name}/` - Source code
- `tests/{project_name}/` - Unit tests  
- `notebooks/{project_name}/` - Jupyter notebooks with visualizations
- `datasets/` - Raw data files

## Build, Test, and Lint Commands

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (linting, testing, type checking)
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific project
pytest tests/naive_bayes_demo/

# Run a single test by name
pytest -k "test_model_predict"
```

### Linting and Code Quality

```bash
# Run ruff linter (auto-fixes most issues)
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Format with ruff
ruff format .

# Run type checker
mypy src/
```

### Jupyter Notebooks

```bash
# Install notebook kernel for testing
pip install ipykernel nbval

# Run notebook tests (ensures notebooks are reproducible)
pytest --nbval notebooks/naive_bayes_demo/naive_bayes_demo.ipynb
```

## Code Style Guidelines

### General Principles

- Write clean, readable, production-quality code
- Every function requires a type-annotated return type
- Use docstrings for all public functions and classes
- Keep functions small and focused (single responsibility)
- No magic numbers - use named constants

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `data_loader.py` |
| Classes | `PascalCase` | `NaiveBayesClassifier` |
| Functions | `snake_case` | `load_dataset()` |
| Variables | `snake_case` | `X_train`, `y_pred` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_ITERATIONS` |
| Private members | Leading underscore | `_private_method()` |

### Import Organization

Imports must be organized in the following order (use `ruff` to enforce):

1. Standard library imports
2. Third-party imports
3. Local application imports

Within each group, sort alphabetically. Example:

```python
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.naive_bayes_demo.data import load_breast_cancer
```

### Type Annotations

- All function parameters must have type annotations
- All function returns must have type annotations
- Use `Optional[X]` instead of `X | None`
- Use `Sequence`, `Mapping` for generic collections when appropriate

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> tuple[GaussianNB, dict[str, float]]:
    """Train a Gaussian Naive Bayes classifier."""
    model = GaussianNB(alpha=alpha)
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    return model, metrics
```

### Error Handling

- Use custom exceptions for domain-specific errors
- Catch specific exceptions, never use bare `except:`
- Fail fast with clear error messages
- Log errors appropriately

```python
class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

def train(X: np.ndarray, y: np.ndarray) -> Model:
    if len(X) == 0 or len(y) == 0:
        raise ModelTrainingError("X and y must not be empty")
    if len(X) != len(y):
        raise ModelTrainingError(f"X and y must have same length: {len(X)} != {len(y)}")
    # ... training logic
```

### ML/AI Specific Guidelines

1. **Reproducibility**: Set random seeds for all stochastic operations
2. **Data Splitting**: Always use stratified splits for classification
3. **Metrics**: Report multiple metrics (accuracy, precision, recall, F1, AUC)
4. **Visualization**: Use consistent color schemes and figure sizes
5. **Notebooks**: All notebooks must be executable top-to-bottom without errors

### File Organization Per Project

Each project directory must contain:

```
{project_name}/
├── __init__.py          # Public API exports
├── data.py              # Dataset loading and preprocessing
├── model.py             # Model implementation/definition
├── train.py             # Training logic
├── evaluate.py          # Evaluation and visualization
└── config.py            # Constants and configuration
```

The notebook filename must match the project name: `{project_name}.ipynb`

### Documentation

- All public classes and functions require docstrings
- Use Google-style docstrings
- Include:
  - Short description
  - Args (if any)
  - Returns (if any)
  - Raises (if applicable)
  - Example usage (for complex functions)

```python
def calculate_prior_probabilities(
    y: np.ndarray,
) -> dict[int, float]:
    """Calculate prior probabilities for each class.
    
    Args:
        y: Array of class labels.
        
    Returns:
        Dictionary mapping class labels to their prior probabilities.
    """
    # implementation
```

### Git Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- First line: max 50 characters
- Body: wrap at 72 characters
- Reference issues when applicable

## Testing Guidelines

- Test files mirror source structure: `tests/{project}/{module}_test.py`
- Use descriptive test names: `test_model_predict_returns_correct_shape`
- Use fixtures for shared setup
- Test edge cases and error conditions
- Aim for >80% code coverage on business logic

## Pre-commit Hooks

Install pre-commit to run checks before each commit:

```bash
pip install pre-commit
pre-commit install
```

This runs ruff, mypy, and other checks automatically.
