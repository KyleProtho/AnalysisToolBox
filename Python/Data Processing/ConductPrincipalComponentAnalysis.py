# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Delcare function
def ConductPrincipalComponentAnalysis(dataframe,
                                      list_of_numeric_variables=None,
                                      number_of_components=None,
                                      random_seed=412):
    # If list_of_numeric_variables is not specified, then use all numeric variables
    if list_of_numeric_variables is None:
        list_of_numeric_variables = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
    # Select only numeric variables
    dataframe_pca = dataframe[list_of_numeric_variables].copy()
    
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
    print(data_pca)

    # Add principal components to original dataframe
    for i in range(1, number_of_components+1):
        dataframe['PC' + str(i)] = data_pca.iloc[i-1, 0:len(dataframe.columns)].dot(dataframe_pca.transpose())

    # Return dataframe
    return(dataframe)


# # Test function
# dataset = pd.read_csv("C:/Users/oneno/OneDrive/Documents/Continuing Education/Udemy/Data Mining for Business in Python/5. Dimension Reduction/abalone-challenge.csv")
# # dataset = ConductPrincipalComponentAnalysis(
# #     dataframe=dataset
# # )

# # Randomly remove 10% of values from the dataset
# dataset = dataset.mask(np.random.random(dataset.shape) < .1)
# dataset = ConductPrincipalComponentAnalysis(
#     dataframe=dataset
# )
