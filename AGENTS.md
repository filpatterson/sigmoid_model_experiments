# AGENTS.md - Sigmoid Model Experiments

## Project Overview

This repository contains classification and regression model implementations using scikit-learn.
Each project is organized in its own directory:
- `src/{project_name}/` - Source code
- `tests/{project_name}/` - Unit tests
- `notebooks/{project_name}/` - Jupyter notebooks
- `datasets/` - Raw data files

## Build, Test, and Lint Commands

### Install Dependencies

**Conda Environment**: `sigmoid_experiments` (Python 3.11)
```bash
conda activate sigmoid_experiments

pip install -r requirements.txt        # Core dependencies
pip install -r requirements-dev.txt   # Dev: pytest, ruff, mypy, pre-commit
```

### Running Tests
```bash
pytest                          # All tests
pytest tests/naive_bayes_demo/  # Tests for specific project
pytest -k "test_model_predict"  # Single test by name
pytest --nbval notebook.ipynb   # Notebook validation
```

### Linting and Formatting
```bash
ruff check .        # Lint (auto-fixes: ruff check --fix .)
ruff format .       # Format code
mypy src/           # Type check
pre-commit install  # Install git hooks
```

## Code Style Guidelines

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `data_loader.py` |
| Classes | `PascalCase` | `NaiveBayesClassifier` |
| Functions/variables | `snake_case` | `load_dataset()`, `X_train` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_ITERATIONS` |
| Private members | Leading underscore | `_private_method()` |

### Import Order (enforced by ruff)
1. Standard library (`logging`, `pathlib`, `typing`)
2. Third-party (`numpy`, `pandas`, `sklearn`)
3. Local application (`src.project.module`)

### Type Annotations
- All function parameters and returns must be annotated
- Use `Optional[X]` not `X | None`
- Use generic collections: `Sequence`, `Mapping`

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> tuple[GaussianNB, dict[str, float]]:
    """Train a Gaussian Naive Bayes classifier."""
```

### Error Handling
- Use custom exceptions for domain errors
- Catch specific exceptions, never bare `except:`
- Fail fast with clear messages

```python
class ModelTrainingError(Exception):
    """Raised when model training fails."""

def train(X: np.ndarray, y: np.ndarray) -> Model:
    if len(X) == 0:
        raise ModelTrainingError("X must not be empty")
```

### ML/AI Specific Guidelines
1. **Reproducibility**: Set random seeds for all stochastic operations
2. **Data Splitting**: Always use stratified splits for classification
3. **Metrics**: Report accuracy, precision, recall, F1, AUC
4. **Notebooks**: Must execute top-to-bottom without errors

## File Organization Per Project

```
{project_name}/
├── __init__.py    # Public API exports
├── data.py        # Dataset loading/preprocessing
├── model.py       # Model definition
├── train.py       # Training logic
├── evaluate.py    # Evaluation/visualization
└── config.py      # Constants/configuration
```

## Documentation

- All public classes/functions require Google-style docstrings
- Include: short description, Args, Returns, Raises (if applicable)

```python
def calculate_prior_probabilities(y: np.ndarray) -> dict[int, float]:
    """Calculate prior probabilities for each class.
    
    Args:
        y: Array of class labels.
        
    Returns:
        Dictionary mapping class labels to their prior probabilities.
    """
```

## Testing Guidelines

- Test files: `tests/{project}/{module}_test.py`
- Test names: `test_model_predict_returns_correct_shape`
- Use fixtures for shared setup
- Test edge cases and error conditions
- Target >80% coverage on business logic

## Git Commit Messages

- Imperative mood: "Add feature" not "Added feature"
- First line ≤50 characters, body wrap at 72 characters
- Reference issues when applicable

## Code Quality Tools

| Tool | Purpose | Config |
|------|---------|--------|
| ruff | Lint + format | `pyproject.toml` |
| mypy | Type checking | `pyproject.toml` |
| pytest | Testing | `pyproject.toml` |
| pre-commit | Git hooks | `.pre-commit-config.yaml` |

### ruff Configuration (from pyproject.toml)
- Line length: 88
- Python target: 3.10
- Ignores E501 (line length, handled by formatter)
- quote-style: double, indent-style: space

### mypy Configuration
- `disallow_untyped_defs = true`
- Third-party libs ignored: sklearn, matplotlib, seaborn, numpy, pandas
