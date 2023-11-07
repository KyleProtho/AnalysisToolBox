# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Declare function
def CreateKMeansClusters(dataframe,
                         list_of_numeric_columns_for_clustering=None,
                         number_of_clusters=None,
                         column_name_for_clusters='K-Means Cluster',
                         scale_predictor_variables=True,
                         show_cluster_summary_plots=True,
                         summary_plot_size=(20, 20),
                         random_seed=412,
                         maximum_iterations=300):
    """
    This function creates K-Means clusters on a dataset based on the variables specified.

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

    # If number of clusters not specified, use elbow to find "best" number
    if number_of_clusters == None:
        model = KMeans()
        visualizer = KElbowVisualizer(
            model,
            k=(2,30),
            timings=True
        )
        visualizer.fit(dataframe)
        visualizer.show()
        number_of_clusters = visualizer.elbow_value_
        
    # Conduct k-means clusters
    model = KMeans(
        n_clusters=number_of_clusters, 
        random_state=random_seed,
        max_iter=maximum_iterations,
    )
    model = model.fit(dataframe_clusters[list_of_numeric_columns_for_clustering])

    # Join clusters to original dataset
    dataframe_clusters[column_name_for_clusters] = model.labels_

    # Convert cluster to categorical data type
    dataframe_clusters[column_name_for_clusters] = dataframe_clusters[column_name_for_clusters].apply(str)
    
    # Join clusters to original dataset
    dataframe_clusters = dataframe_clusters[[column_name_for_clusters]]
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
            hue=column_name_for_clusters
        )
        plt.suptitle("Cluster Summary Plots", fontsize=15)
        plt.show()

    # Return updated dataset
    return(dataframe)

