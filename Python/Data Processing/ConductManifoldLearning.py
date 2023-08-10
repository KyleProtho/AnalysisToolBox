# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# Delcare function
def ConductManifoldLearning(dataframe,
                            list_of_numeric_variables=None,
                            number_of_components=3,
                            random_seed=412,
                            show_component_summary_plots=True,
                            summary_plot_size=(20, 20)):
    # If list_of_numeric_variables is not specified, then use all numeric variables
    if list_of_numeric_variables is None:
        list_of_numeric_variables = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
    # Select only numeric variables
    dataframe_manifold = dataframe[list_of_numeric_variables].copy()
    
    # Remove missing values
    dataframe_manifold = dataframe_manifold.dropna()

    # Create list of column names
    list_of_column_names = dataframe_manifold.columns.tolist()

    # Conduct Manifold Learning
    model = TSNE(
        n_components=number_of_components,
        random_state=random_seed
    )
    components = model.fit_transform(dataframe_manifold)
    
    # Get column names for new components
    list_of_component_names = []
    for i in range(1, number_of_components + 1):
        list_of_component_names.append("MLC" + str(i))
    
    # Change component column names
    components = pd.DataFrame(
        data=components,
        columns=list_of_component_names
    )

    # Add Manifold Learning components to original dataframe
    dataframe = pd.concat([dataframe, components], axis=1)
    
    # If requested, show box plots of each component for each variable
    # Put each numeric variables on the Y axis and each component on the X axis
    if show_component_summary_plots:
        plt.figure(figsize=summary_plot_size)
        sns.pairplot(
            data=dataframe[list_of_numeric_variables + list_of_component_names],
            x_vars=list_of_component_names,
            y_vars=list_of_numeric_variables,
            kind='kde'
        )
        plt.suptitle("Component Summary Plots", fontsize=15)
        plt.show()
    
    # Return dataframe
    return(dataframe)


# # Test function
# dataset = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/5. Dimension Reduction/abalone-challenge.csv")
# # dataset = ConductManifoldLearning(
# #     dataframe=dataset
# # )
# # Randomly remove 10% of values from the dataset
# dataset = dataset.mask(np.random.random(dataset.shape) < .1)
# # Conduct Manifold Learning
# dataset = ConductManifoldLearning(
#     dataframe=dataset
# )
