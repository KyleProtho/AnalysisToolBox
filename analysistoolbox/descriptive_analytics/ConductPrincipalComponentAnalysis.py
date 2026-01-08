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
    Perform Principal Component Analysis (PCA) and return enriched DataFrame with components.

    This function applies Principal Component Analysis to reduce dimensionality while retaining
    maximum variance in the data. PCA transforms correlated variables into a set of uncorrelated
    principal components ordered by the amount of variance they explain. The function automatically
    scales features using MinMaxScaler and can intelligently determine the optimal number of
    components needed to capture 90% of the total variance.

    Principal Component Analysis is essential for:
      * Dimensionality reduction while preserving data variance
      * Feature engineering and noise reduction in machine learning
      * Multicollinearity elimination in regression analysis
      * Data compression and storage optimization
      * Exploratory data analysis and pattern recognition
      * Visualization of high-dimensional data
      * Computational efficiency improvement for downstream algorithms
      * Identifying the most important directions of variation in data

    When the number of components is not specified, the function generates a scree plot showing
    explained variance for each component and automatically selects the minimum number needed
    to capture ≥90% of total variance. The principal component loadings are displayed as a
    table, showing how each original feature contributes to each component.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to transform. Must include numeric columns
        for the PCA algorithm to process. Rows with missing values are automatically removed.
    list_of_numeric_columns
        List of column names to include in the PCA. If None, all numeric columns in the
        DataFrame will be automatically selected. Defaults to None.
    number_of_components
        Number of principal components to retain. If None, the function automatically
        determines the optimal number by identifying components that cumulatively explain
        ≥90% of variance. Defaults to None.
    random_seed
        Random seed for reproducibility of the PCA algorithm. Use the same seed to ensure
        consistent results across multiple runs. Defaults to 412.
    display_pca_as_markdown
        Whether to display the principal component loadings table in the output. The table
        shows how each original feature contributes to each principal component. Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional columns for each principal component.
        New columns are named 'PC1', 'PC2', ..., 'PCn' where n is the number_of_components.
        Rows with missing values in the selected numeric columns are removed.

    Examples
    --------
    # Automatic component selection with variance threshold
    import pandas as pd
    marketing_df = pd.DataFrame({
        'ad_spend': [1000, 2000, 1500, 3000, 2500],
        'impressions': [10000, 20000, 15000, 30000, 25000],
        'clicks': [100, 200, 150, 300, 250],
        'conversions': [10, 20, 15, 30, 25]
    })
    result_df = ConductPrincipalComponentAnalysis(
        marketing_df,
        display_pca_as_markdown=True
    )
    # Automatically selects components explaining ≥90% variance
    # Displays scree plot and component loadings

    # Manual specification of components for specific use case
    feature_df = pd.DataFrame({
        'temperature': [20, 25, 30, 22, 28],
        'humidity': [60, 65, 70, 62, 68],
        'pressure': [1013, 1015, 1012, 1014, 1016],
        'wind_speed': [10, 12, 8, 11, 9]
    })
    pca_df = ConductPrincipalComponentAnalysis(
        feature_df,
        number_of_components=2,
        random_seed=42,
        display_pca_as_markdown=False
    )
    # Returns DataFrame with PC1 and PC2 columns added

    # Selecting specific columns for PCA analysis
    sales_df = pd.DataFrame({
        'product_price': [10, 20, 15, 25, 30],
        'units_sold': [100, 80, 90, 70, 60],
        'revenue': [1000, 1600, 1350, 1750, 1800],
        'customer_id': ['A', 'B', 'C', 'D', 'E']
    })
    reduced_df = ConductPrincipalComponentAnalysis(
        sales_df,
        list_of_numeric_columns=['product_price', 'units_sold', 'revenue'],
        number_of_components=2,
        random_seed=412
    )
    # PCA applied only to specified numeric columns

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

