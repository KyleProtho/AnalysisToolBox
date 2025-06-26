# Load packages
import numpy as np
import pandas as pd
from analysistoolbox.simulations import CreateCorrelatedSIPs

# Use the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Calculate correlation
correlation = data['sepal length (cm)'].corr(data['sepal width (cm)'])

# Create correlated SIPs
CreateCorrelatedSIPs(
    data, 
    mean_of_variable_1=data['sepal length (cm)'].mean(),
    std_of_variable_1=data['sepal length (cm)'].std(),
    mean_of_variable_2=data['sepal width (cm)'].mean(),
    std_of_variable_2=data['sepal width (cm)'].std(),
    correlation=correlation,
    number_of_samples=10000,
)
