# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis Tool Box is a comprehensive Python package for data collection, processing, statistics, analytics, and intelligence analysis. The package provides 100+ functions across 14 specialized modules, designed for analysts, data scientists, and researchers.

## Development Commands

### Package Building and Testing
- **Build package**: `python setup.py sdist bdist_wheel`
- **Install for development**: `pip install -e .`
- **Run tests**: `python -m pytest` or `python -m unittest`
- **Install from PyPI**: `pip install analysistoolbox`

### CI/CD Pipeline
- **Automated deployment**: GitHub Actions workflow in `.github/workflows/python-publish.yml`
- **Trigger**: Releases automatically deploy to PyPI when published
- **Version management**: Update version in `setup.py` following semantic versioning

## Code Architecture

### Module Structure
The codebase follows a strict modular architecture with 14 main functional areas:

- **calculus/**: Mathematical functions and derivatives
- **data_collection/**: Web scraping, PDF extraction, API integrations  
- **data_processing/**: Data cleaning, transformation, and preprocessing
- **descriptive_analytics/**: Clustering, PCA, manifold learning
- **file_management/**: File operations and conversions
- **hypothesis_testing/**: Statistical tests and regression analysis
- **linear_algebra/**: Matrix operations and linear transformations
- **llm/**: Integration with Anthropic Claude and OpenAI APIs
- **predictive_analytics/**: Machine learning models
- **prescriptive_analytics/**: Recommendation systems
- **probability/**: Probability calculations and distributions
- **simulations/**: Statistical simulations and distributions
- **statistics/**: Statistical inference and confidence intervals
- **visualizations/**: Publication-quality charts and plots

### Function Design Patterns

Each module follows consistent patterns:

1. **One function per file**: Each Python file contains a single function with the same name as the file
2. **PascalCase naming**: Function names use PascalCase (e.g., `FindDerivative`, `PlotBarChart`)
3. **Module imports**: Each module's `__init__.py` imports all functions for clean API access
4. **Lazy loading**: Heavy dependencies are imported inside functions for performance

### Standard Function Structure
```python
def FunctionName(required_params,
                 optional_param=default_value,
                 # Grouped optional parameters with comments
                 plot_related_param=True,
                 # Plot formatting arguments
                 figure_size=(8, 6),
                 # Text formatting arguments  
                 title_for_plot=None,
                 # Plot saving arguments
                 filepath_to_save_plot=None):
    """
    Comprehensive docstring with Args and Returns sections.
    """
    # Lazy load dependencies
    import heavy_library
    
    # Parameter validation with descriptive errors
    # Core functionality implementation
    # Optional plotting capabilities
    # Return data, plots, or both
```

### Key Architectural Decisions

1. **Single Responsibility**: Each function has one clear purpose
2. **Comprehensive Parameters**: Extensive customization options with sensible defaults
3. **Integrated Visualization**: Optional plotting built into analysis functions
4. **Flexible Returns**: Functions can return data, plots, or both based on parameters
5. **Error Handling**: Descriptive error messages with suggested fixes
6. **Performance Optimization**: Lazy loading of dependencies

## Dependencies and Requirements

### Core Dependencies (70+ packages)
Key libraries include pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, xgboost, sympy, statsmodels, langchain_anthropic, openai, geopandas, beautifulsoup4, and more.

### Development Dependencies
- `pytest>=3.7` for testing
- `twine>=4.0.2` for PyPI publishing

### Python Requirements
- **Minimum version**: Python 3.9+
- **Platform**: Cross-platform (Windows, macOS, Linux)

## Testing Strategy

### Test Structure
- **Location**: `/analysistoolbox/tests/`
- **Pattern**: `test_FunctionName.py` files
- **Framework**: unittest
- **Coverage**: Unit tests for core functions

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test
python -m unittest analysistoolbox.tests.test_FunctionName

# Development install for testing
pip install -e .
```

## Development Guidelines

### Adding New Functions

1. **Create function file**: `analysistoolbox/module_name/FunctionName.py`
2. **Follow naming**: Use PascalCase for function names
3. **Update module init**: Add import to `analysistoolbox/module_name/__init__.py`
4. **Add dependencies**: Update `setup.py` if new packages are required
5. **Write tests**: Create corresponding test file
6. **Update documentation**: Add usage example to README.md

### Code Quality Standards

1. **Documentation**: Every function must have comprehensive docstrings
2. **Parameter validation**: Include input validation with descriptive errors
3. **Lazy loading**: Import heavy dependencies inside functions
4. **Consistent patterns**: Follow existing parameter naming and structure
5. **Error handling**: Provide helpful error messages and suggested fixes

### Version Management

- **Current version**: 2.2.85 (indicates mature, active development)
- **Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Release process**: Update setup.py version → create GitHub release → auto-deploy via GitHub Actions

## Integration Capabilities

The package integrates with major analytics and ML ecosystems:

- **LLM APIs**: Anthropic Claude, OpenAI ChatGPT
- **Geospatial**: geopandas, folium for mapping
- **ML Frameworks**: TensorFlow, XGBoost, scikit-learn
- **Statistical**: statsmodels, lifelines for survival analysis
- **Visualization**: matplotlib, seaborn, plotly-style outputs

## Performance Considerations

1. **Lazy loading**: Heavy imports happen only when needed
2. **Optional plotting**: Visualization libraries loaded only when plotting
3. **Memory efficiency**: Large operations use chunking where appropriate
4. **Caching**: Some functions implement result caching for expensive operations

## Common Development Tasks

When adding new functionality:

1. **Research existing patterns**: Check similar functions for consistent parameter naming
2. **Follow module conventions**: Each module has specific patterns for its domain
3. **Test thoroughly**: Create unit tests and manual testing scenarios
4. **Document comprehensively**: Include usage examples in both docstrings and README
5. **Consider performance**: Use lazy loading for optional dependencies
6. **Maintain backwards compatibility**: Avoid breaking changes in public APIs