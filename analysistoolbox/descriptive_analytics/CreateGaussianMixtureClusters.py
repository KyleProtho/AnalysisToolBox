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
                                  scale_predictor_variables=False,
                                  show_cluster_summary_plots=True,
                                  sns_color_palette='Set1',
                                  summary_plot_size=(20, 20),
                                  random_seed=412,
                                  maximum_iterations=300):
    """
    This function creates Gaussian Mixture clusters on a dataset based on the variables specified.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        list_of_numeric_columns_for_clustering (list, optional): The list of variables to base the clusters on. Defaults to None, which will use all variables in the dataframe.
        number_of_clusters (int, optional): The number of clusters to create. Defaults to None, which will use the elbow method to determine the optimal number of clusters.
        column_name_for_clusters (str, optional): The name of the new column containing the clusters. Defaults to 'K-Means Cluster'.
        scale_predictor_variables (bool, optional): Whether to scale the predictor variables prior to analysis. Defaults to True.
        show_cluster_summary_plots (bool, optional): Whether to show cluster summary plots. Defaults to True.
        summary_plot_size (tuple, optional): The size of the summary plots. Defaults to (20, 20).
        random_seed (int, optional): The random seed to use for replication. Defaults to 412.
        maximum_iterations (int, optional): The maximum number of iterations to use for the K-Means algorithm. Defaults to 300.
    
    Returns:
        Pandas dataframe: An updated Pandas dataframe with the clusters joined to the original data.
    """
    
    # Keep complete cases only
    dataframe_clusters = dataframe.dropna(subset=list_of_numeric_columns_for_clustering)

    # Scale the predictors, if requested
    if scale_predictor_variables:
        # Scale predictors
        dataframe_clusters[list_of_numeric_columns_for_clustering] = StandardScaler().fit_transform(dataframe_clusters[list_of_numeric_columns_for_clustering])
    
    # Show peak-to-peak range of each predictor
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

