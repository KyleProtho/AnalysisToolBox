# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Set arguments
dataframe = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/5. Dimension Reduction/abalone-challenge.csv")
list_of_numeric_variables=None

# If list_of_numeric_variables is not specified, then use all numeric variables
if list_of_numeric_variables is None:
    list_of_numeric_variables = dataframe.select_dtypes(include=[np.number]).columns.tolist()

