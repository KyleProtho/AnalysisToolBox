# Load packages
from IPython.display import display, HTML, Markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Delcare function
def ConductPrincipalComponentAnalysis(dataframe,
                                      list_of_numeric_columns=None,
                                      number_of_components=None,
                                      random_seed=412,
                                      display_pca_as_markdown=True):
    """
    Conducts principal component analysis on a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe to perform PCA on.
        list_of_numeric_columns (list, optional): A list of column names to use for PCA. If not specified, all numeric columns will be used. Defaults to None.
        number_of_components (int, optional): The number of principal components to use. If not specified, the function will determine the optimal number of components. Defaults to None.
        random_seed (int, optional): The random seed to use for reproducibility. Defaults to 412.
        display_pca_as_markdown (bool, optional): Whether to display the principal components as a markdown table. Defaults to True.

    Returns:
        pandas.DataFrame: The original dataframe with the principal components added.
    """
    
    # If list_of_numeric_columns is not specified, then use all numeric variables
    if list_of_numeric_columns is None:
        list_of_numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
    # Select only numeric variables
    dataframe_pca = dataframe[list_of_numeric_columns].copy()
    
    # Remove missing values
    dataframe_pca = dataframe_pca.dropna()

    # Create list of column names
    list_of_column_names = dataframe_pca.columns.tolist()
        
    # Scale the data
    scaler = MinMaxScaler()
    dataframe_pca = pd.DataFrame(scaler.fit_transform(dataframe_pca))
    dataframe_pca.columns = list_of_column_names

    # Determine optimal number of components
    if number_of_components is None:
        model = PCA(random_state=random_seed).fit(dataframe_pca)
        plt.plot(model.explained_variance_ratio_)
        plt.ylabel('Explained Variance')
        plt.show()
        
        # Determine number of components to use
        for i in range(1, len(model.explained_variance_ratio_)):
            if sum(model.explained_variance_ratio_[0:i]) >= 0.90:
                number_of_components = i
                break
        print("Number of components to use: " + str(number_of_components))
        print("Total variance explained: " + str(round(sum(model.explained_variance_ratio_[0:i])*100, 2)) + "%")
        print("Be sure to review the scree plot to ensure that the number of components selected is appropriate for your purpose.")

    # Conduct PCA
    model = PCA(n_components=number_of_components,
                random_state=random_seed).fit(dataframe_pca)

    # Create dataframe of principal components
    data_pca = pd.DataFrame(
        model.components_,
        columns=dataframe_pca.columns
    )
    data_pca.index = ['PC' + str(i) for i in range(1, number_of_components+1)]

    # Print principal components
    print("\nPrincipal components:")
    if display_pca_as_markdown:
        display(data_pca)
    else:
        print(data_pca)

    # Add principal components to original dataframe
    for i in range(1, number_of_components+1):
        dataframe['PC' + str(i)] = data_pca.iloc[i-1, 0:len(dataframe.columns)].dot(dataframe_pca.transpose())

    # Return dataframe
    return(dataframe)

