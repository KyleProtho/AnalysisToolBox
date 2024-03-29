# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# Delcare function
def ConductManifoldLearning(dataframe,
                            list_of_numeric_columns=None,
                            number_of_components=3,
                            random_seed=412,
                            show_component_summary_plots=True,
                            summary_plot_size=(20, 20)):
    """
    Conducts manifold learning on a given dataframe and returns a new dataframe with the
    original columns and the new manifold learning components.

    Args:
        dataframe (pandas.DataFrame): The input dataframe.
        list_of_numeric_columns (list, optional): A list of column names to use for the manifold
            learning. If None, all numeric columns will be used. Defaults to None.
        number_of_components (int, optional): The number of components to generate. Defaults to 3.
        random_seed (int, optional): The random seed to use for the manifold learning algorithm.
            Defaults to 412.
        show_component_summary_plots (bool, optional): Whether to show summary plots of each
            component for each variable. Defaults to True.
        summary_plot_size (tuple, optional): The size of the summary plots. Defaults to (20, 20).

    Returns:
        pandas.DataFrame: A new dataframe with the original columns and the new manifold learning
        components.
    """
    
    # If list_of_numeric_columns is not specified, then use all numeric variables
    if list_of_numeric_columns is None:
        list_of_numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
    # Select only numeric variables
    dataframe_manifold = dataframe[list_of_numeric_columns].copy()
    
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
            data=dataframe[list_of_numeric_columns + list_of_component_names],
            x_vars=list_of_component_names,
            y_vars=list_of_numeric_columns,
            kind='kde'
        )
        plt.suptitle("Component Summary Plots", fontsize=15)
        plt.show()
    
    # Return dataframe
    return(dataframe)

