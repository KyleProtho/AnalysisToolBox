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
    Perform t-SNE manifold learning dimensionality reduction and return enriched DataFrame.

    This function applies t-distributed Stochastic Neighbor Embedding (t-SNE) to reduce
    high-dimensional data into a lower-dimensional space while preserving local structure
    and relationships. t-SNE is particularly effective at revealing clusters and patterns
    in complex datasets by mapping similar data points close together in the reduced space.

    Manifold learning with t-SNE is essential for:
      * Exploratory data analysis and pattern discovery in high-dimensional datasets
      * Visualizing complex data structures in 2D or 3D space
      * Feature engineering for machine learning pipelines
      * Identifying clusters and outliers in multivariate data
      * Reducing computational complexity for downstream analysis
      * Creating interpretable representations of complex relationships
      * Data preprocessing for classification and clustering tasks

    The function automatically handles missing values by removing rows with NaN entries
    and generates new component columns named 'MLC1', 'MLC2', etc. Optional visualization
    using pair plots helps interpret the relationship between original features and the
    learned manifold components.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to transform. Must include numeric columns
        for the manifold learning algorithm to process.
    list_of_numeric_columns
        List of column names to use for manifold learning. If None, all numeric columns
        in the DataFrame will be automatically selected. Defaults to None.
    number_of_components
        Number of dimensions in the reduced manifold space. Common values are 2 or 3 for
        visualization purposes. Defaults to 3.
    random_seed
        Random seed for reproducibility of the t-SNE algorithm. Use the same seed to
        ensure consistent results across multiple runs. Defaults to 412.
    show_component_summary_plots
        Whether to display pair plots showing the relationship between original features
        and the learned manifold components using kernel density estimation. Defaults to True.
    summary_plot_size
        Figure size for the summary pair plots as a tuple of (width, height) in inches.
        Defaults to (20, 20).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional columns for each manifold learning component.
        New columns are named 'MLC1', 'MLC2', ..., 'MLCn' where n is the number_of_components.
        Rows with missing values in the selected numeric columns are removed.

    Examples
    --------
    # Reduce high-dimensional customer data to 2D for visualization
    import pandas as pd
    customer_df = pd.DataFrame({
        'age': [25, 34, 45, 23, 56],
        'income': [50000, 75000, 90000, 45000, 120000],
        'spending_score': [60, 81, 45, 72, 38],
        'loyalty_years': [1, 5, 10, 2, 15]
    })
    result_df = ConductManifoldLearning(
        customer_df,
        number_of_components=2,
        show_component_summary_plots=False
    )
    # Result includes: MLC1, MLC2 columns for clustering analysis

    # Analyze specific features with visualization
    feature_df = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200),
        'feature3': range(200, 300),
        'category': ['A'] * 50 + ['B'] * 50
    })
    manifold_df = ConductManifoldLearning(
        feature_df,
        list_of_numeric_columns=['feature1', 'feature2', 'feature3'],
        number_of_components=3,
        random_seed=42,
        show_component_summary_plots=True,
        summary_plot_size=(15, 15)
    )

    # Create 3D manifold representation for complex dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=20, n_informative=15)
    complex_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    complex_df['target'] = y
    reduced_df = ConductManifoldLearning(
        complex_df,
        number_of_components=3,
        random_seed=412
    )
    # Use MLC1, MLC2, MLC3 for 3D visualization or further modeling

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

