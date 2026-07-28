# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`analysistoolbox` is a Python package (v3.9.92) for data collection, processing, statistics, analytics, and intelligence analysis. It is distributed via PyPI.

## Commands

**Install for development:**
```sh
pip install -e ".[dev]"
```

**Run all tests:**
```sh
python -m unittest discover -s analysistoolbox/tests
```

**Run a single test file:**
```sh
python -m unittest analysistoolbox.tests.test_PlotBarChart
```

**Build for distribution:**
```sh
python setup.py sdist bdist_wheel
```

**Upload to PyPI:**
```sh
python -m twine upload dist/*
```

## Architecture

### One Function Per File (PascalCase)
Every function lives in its own file named identically to the function (e.g., `FindDerivative.py` exports `FindDerivative`). File and function names use PascalCase — this is intentional and departs from PEP8 convention.

### Module Structure
Each submodule (e.g., `calculus`, `visualizations`) has an `__init__.py` that explicitly re-exports all its functions. Users import like:
```python
from analysistoolbox.calculus import FindDerivative
from analysistoolbox.visualizations import PlotBarChart
```

### Submodules
| Module | Contents |
|---|---|
| `calculus` | Derivatives, limits, optimization |
| `data_collection` | Web scraping, PDF extraction, API calls |
| `data_processing` | Cleaning, transformation, outlier detection |
| `descriptive_analytics` | Clustering, PCA, manifold learning |
| `file_management` | PDF/document handling, file trees |
| `geospatial_analysis` | Spatial clustering, autocorrelation |
| `hypothesis_testing` | Statistical tests, regression, ANOVA |
| `linear_algebra` | Matrix operations, eigenvalues |
| `llm` | Anthropic and OpenAI API integration |
| `network_analysis` | Graph construction from edge lists, network metrics |
| `predictive_analytics` | ARIMA, XGBoost, neural networks |
| `prescriptive_analytics` | Linear optimization, recommenders |
| `probability` | Probability utilities |
| `simulations` | Monte Carlo, metalog/SLURP distributions |
| `statistics` | Confidence intervals, descriptive stats |
| `visualizations` | 19 chart/plot functions |

### Tests
Tests use `unittest.TestCase` and live in `analysistoolbox/tests/`. Visualization tests must call `plt.clf()` in `tearDown()`.

## Docstring Requirements

Every function **must** include a "Teaching Note" section in its docstring explaining *why* the technique matters analytically — not just how to use it. This is a core project requirement. Follow NumPy docstring style.

## Versioning & Release

Update the version in `setup.py` following semantic versioning before each release. The GitHub Action at `.github/workflows/python-publish.yml` publishes to PyPI on new tags.
