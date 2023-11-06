# Analysis Tool Box

## Description

Analysis Tool Box (i.e. "analysistoolbox") is a collection of tools in Python for data collection and processing, statisitics, analytics, and intelligence analysis.

## Getting Started

To install the package, run the following command in the root directory of the project:

```bash
pip install analysistoolbox
```

## Usage

There are many submodules in the AnalysisToolBox package, each with their own functions. The following is a list of the submodules:

* Calculus
* DataCollection
* DataProcessing
* DescriptiveAnalytics
* FileManagement
* HypothesisTesting
* LinearAlgebra
* Statistics
* Visualizations

### Calculus

#### FindDerivative

The **FindDerivative** function calculates the derivative of a given function. It uses the sympy library, a Python library for symbolic mathematics, to perform the differentiation. The function also has the capability to print the original function and its derivative, return the derivative function, and plot both the original function and its derivative.

```python
# Load the FindDerivative function from the Calculus submodule
from analysistoolbox.calculus import FindDerivative
import sympy

# Define a symbolic variable
x = sympy.symbols('x')

# Define a function
f_of_x = x**3 + 2*x**2 + 3*x + 4

# Use the FindDerivative function
FindDerivative(
    f_of_x, 
    print_functions=True, 
    return_derivative_function=True, 
    plot_functions=True
)
```

#### FindLimitOfFunction

The **FindLimitOfFunction** function finds the limit of a function at a specific point and optionally plot the function and its tangent line at that point. The script uses the matplotlib and numpy libraries for plotting and numerical operations respectively.

```python
# Import the necessary libraries
import numpy as np
from analysistoolbox.calculus import FindLimitOfFunction

# Define a function
f_of_x = lambda x: np.sin(x) / x

# Use the FindLimitOfFunction function
FindLimitOfFunction(
    f_of_x, 
    point=0, 
    step=0.01, 
    plot_function=True, 
    x_minimum=-10, 
    x_maximum=10, 
    n=1000, 
    tangent_line_window=1
)
```

#### FindMinimumSquareLoss

The **FindMinimumSquareLoss** function calculates the minimum square loss between observed and predicted values. This function is often used in machine learning and statistics to measure the average squared difference between the actual and predicted outcomes.

```python
# Import the necessary libraries
from analysistoolbox.calculus import FindMinimumSquareLoss

# Define observed and predicted values
observed_values = [1, 2, 3, 4, 5]
predicted_values = [1.1, 1.9, 3.2, 3.7, 5.1]

# Use the FindMinimumSquareLoss function
minimum_square_loss = FindMinimumSquareLoss(
    observed_values, 
    predicted_values, 
    show_plot=True
)

print(f"The minimum square loss is: {minimum_square_loss}")
```

#### PlotFunction

The **PlotFunction** function plots a mathematical function of x. It takes a lambda function as input and allows for customization of the plot.

```python
# Import the necessary libraries
from analysistoolbox.calculus import PlotFunction

# Define the function to plot
f_of_x = lambda x: x**2

# Plot the function with default settings
PlotFunction(f_of_x)
```

Contributing guidelines: How can people contribute to the project?
License: What license is the project distributed under?