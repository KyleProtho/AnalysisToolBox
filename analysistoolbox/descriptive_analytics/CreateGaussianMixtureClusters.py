# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Declare function
def CreateGaussianMixtureClusters(dataframe,
                                  list_of_numeric_columns_for_clustering=None,
                                  number_of_clusters=None,
                                  column_name_for_clusters='Gaussian Mixture Cluster',
                                  scale_clustering_column_values=False,
                                  # Output arguments
                                  print_peak_to_peak_range_of_each_column=False,
                                  show_cluster_summary_plots=True,
                                  sns_color_palette='Set1',
                                  summary_plot_size=(20, 20),
                                  random_seed=412,
                                  maximum_iterations=300):
    """
    Perform probabilistic clustering using Gaussian Mixture Models (GMM).

    This function applies Gaussian Mixture Model clustering to identify latent groups in data
    by modeling the distribution as a mixture of multiple Gaussian distributions. Unlike hard
    clustering methods like K-Means, GMM provides soft cluster assignments with probabilities,
    allowing each observation to belong to multiple clusters with varying degrees of membership.
    The function automatically determines the optimal number of clusters using BIC/AIC criteria
    when not specified, and provides detailed probability scores for each cluster assignment.

    Gaussian Mixture Model clustering is essential for:
      * Customer segmentation with overlapping behavioral patterns
      * Anomaly detection and outlier identification
      * Market segmentation with fuzzy boundaries between segments
      * Image segmentation and computer vision applications
      * Density estimation and generative modeling
      * Bioinformatics and gene expression analysis
      * Financial risk profiling with probabilistic group membership
      * Natural language processing and topic modeling

    The function generates BIC (Bayesian Information Criterion) and AIC (Akaike Information
    Criterion) plots when the number of clusters is not specified, automatically selecting
    the optimal number based on minimum BIC. It returns cluster assignments along with
    probability scores for each cluster, enabling nuanced interpretation of cluster membership.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to cluster. Rows with missing values in
        clustering columns will be excluded from the analysis.
    list_of_numeric_columns_for_clustering
        List of numeric column names to use as features for clustering. If None, all numeric
        columns in the DataFrame will be used. Defaults to None.
    number_of_clusters
        Number of Gaussian components (clusters) to fit. If None, the function automatically
        determines the optimal number (1-12) using BIC minimization. Defaults to None.
    column_name_for_clusters
        Name for the new column containing cluster assignments (as strings). Defaults to
        'Gaussian Mixture Cluster'.
    scale_clustering_column_values
        Whether to standardize features to zero mean and unit variance before clustering.
        Recommended when features have different scales. Defaults to False.
    print_peak_to_peak_range_of_each_column
        Whether to print the peak-to-peak range of each clustering variable, useful for
        assessing scale differences. Defaults to False.
    show_cluster_summary_plots
        Whether to generate pair plots showing the relationship between clustering variables,
        color-coded by cluster assignment. Defaults to True.
    sns_color_palette
        Seaborn color palette name for cluster visualization. Options include 'Set1', 'Set2',
        'husl', 'colorblind', etc. Defaults to 'Set1'.
    summary_plot_size
        Figure size for the summary pair plots as a tuple of (width, height) in inches.
        Defaults to (20, 20).
    random_seed
        Random seed for reproducibility of the GMM algorithm. Defaults to 412.
    maximum_iterations
        Maximum number of expectation-maximization iterations for model convergence.
        Defaults to 300.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with additional columns:
          * Cluster assignment column: String labels indicating the most likely cluster
          * Probability columns: One column per cluster showing membership probability (0-1)
            named '{column_name_for_clusters}_{i} Probability' where i is the cluster number
        Rows with missing values in clustering columns are excluded and will have NaN in
        the new columns.

    Examples
    --------
    # Customer segmentation with automatic cluster selection
    import pandas as pd
    customer_df = pd.DataFrame({
        'customer_id': range(1, 201),
        'annual_spend': [1000 + i * 50 for i in range(200)],
        'visit_frequency': [5 + i % 30 for i in range(200)],
        'avg_basket_size': [50 + i % 100 for i in range(200)]
    })
    segmented_df = CreateGaussianMixtureClusters(
        customer_df,
        list_of_numeric_columns_for_clustering=['annual_spend', 'visit_frequency', 'avg_basket_size'],
        scale_clustering_column_values=True,
        show_cluster_summary_plots=True
    )
    # Automatically determines optimal clusters and shows probabilities

    # Market segmentation with specified clusters and custom naming
    market_df = pd.DataFrame({
        'household_income': [30000 + i * 1000 for i in range(150)],
        'age': [25 + i % 50 for i in range(150)],
        'education_years': [12 + i % 8 for i in range(150)],
        'family_size': [1 + i % 5 for i in range(150)]
    })
    market_segments = CreateGaussianMixtureClusters(
        market_df,
        list_of_numeric_columns_for_clustering=['household_income', 'age', 'education_years', 'family_size'],
        number_of_clusters=4,
        column_name_for_clusters='Market Segment',
        scale_clustering_column_values=True,
        sns_color_palette='husl',
        random_seed=42
    )
    # Creates 4 market segments with probability scores

    # Anomaly detection using cluster probabilities
    transaction_df = pd.DataFrame({
        'transaction_amount': [100 + i * 10 for i in range(100)],
        'transaction_frequency': [5 + i % 20 for i in range(100)],
        'account_age_days': [30 + i * 5 for i in range(100)]
    })
    clustered_df = CreateGaussianMixtureClusters(
        transaction_df,
        list_of_numeric_columns_for_clustering=['transaction_amount', 'transaction_frequency', 'account_age_days'],
        number_of_clusters=3,
        scale_clustering_column_values=True,
        show_cluster_summary_plots=False,
        print_peak_to_peak_range_of_each_column=True
    )
    # Use low probabilities across all clusters to identify anomalies

    """
    
    # Keep complete cases only
    dataframe_clusters = dataframe.dropna(subset=list_of_numeric_columns_for_clustering)

    # Scale the predictors, if requested
    if scale_clustering_column_values:
        # Scale predictors
        dataframe_clusters[list_of_numeric_columns_for_clustering] = StandardScaler().fit_transform(dataframe_clusters[list_of_numeric_columns_for_clustering])
    
    # Show peak-to-peak range of each predictor
    if print_peak_to_peak_range_of_each_column:
        print("\nPeak-to-peak range of each predictor:")
        print(np.ptp(dataframe_clusters[list_of_numeric_columns_for_clustering], axis=0))
    
    # If number_of_clusters is None, conduct clustering up to 12 times and plot the results
    if number_of_clusters is None:
        n_clusters = np.arange(1,13)
        models = [GaussianMixture(n, random_state=random_seed).fit(dataframe_clusters[list_of_numeric_columns_for_clustering]) for n in n_clusters]
        plt.plot(
            n_clusters,
            [m.bic(dataframe_clusters) for m in models],
            label = 'BIC'
        )
        plt.plot(
            n_clusters,
            [m.aic(dataframe_clusters) for m in models],
            label = 'AIC'
        )
        plt.legend()
        plt.xlabel('Number of Clusters')
        
        # Find the optimal number of clusters based on the BIC and AIC
        number_of_clusters = np.argmin([m.bic(dataframe_clusters) for m in models]) + 1
        print("\nNumber of clusters based on minimum BIC: " + str(number_of_clusters))
        print("But, be sure to check the AIC and BIC curves on the plot to ensure the optimal number of clusters makes sense for your purpose.")
        
    # Conduct Guassian Mixture clustering
    model = GaussianMixture(
        n_components=number_of_clusters, 
        random_state=random_seed,
        max_iter=maximum_iterations
    )
    model = model.fit(dataframe_clusters[list_of_numeric_columns_for_clustering])

    # Get clusters 
    clusters = pd.Series(model.predict(dataframe_clusters[list_of_numeric_columns_for_clustering]))
    dataframe_clusters[column_name_for_clusters] = clusters

    # Convert cluster to categorical data type
    dataframe_clusters[column_name_for_clusters] = dataframe_clusters[column_name_for_clusters].apply(str)
    
    # Get cluster probabilities 
    probabilities = round(pd.DataFrame(model.predict_proba(dataframe_clusters[list_of_numeric_columns_for_clustering])), 4)
    probabilities.columns = [column_name_for_clusters + '_' + str(i) + ' Probability' for i in range(0, number_of_clusters)]
    dataframe_clusters = pd.concat([dataframe_clusters, probabilities], axis = 1)
    
    # Join clusters to original dataset
    dataframe_clusters = dataframe_clusters[[column_name_for_clusters] + probabilities.columns.tolist()]
    dataframe = dataframe.merge(
        dataframe_clusters,
        how='left',
        left_index=True,
        right_index=True
    )
    
    # If requested, show box plots of each cluster for each variable selected
    if show_cluster_summary_plots:
        plt.figure(figsize=summary_plot_size)
        sns.pairplot(
            data=dataframe[list_of_numeric_columns_for_clustering + [column_name_for_clusters]],
            hue=column_name_for_clusters,
            palette=sns_color_palette,
        )
        plt.suptitle("Cluster Summary Plots", fontsize=15)
        plt.show()

    # Return updated dataset
    return(dataframe)

