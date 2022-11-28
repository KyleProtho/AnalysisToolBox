from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Declare function
def CreateKMeansClusters(dataframe,
                         list_of_variables_to_base_clusters=None,
                         number_of_clusters=None,
                         column_name_for_clusters='K-Means Cluster',
                         scale_predictor_variables=True,
                         show_cluster_summary_plots=True,
                         summary_plot_size=(20, 20),
                         random_seed=412,
                         maximum_iterations=300):
    # Keep complete cases only
    dataframe_clusters = dataframe.dropna(subset=list_of_variables_to_base_clusters)

    # Scale the predictors, if requested
    if scale_predictor_variables:
        # Scale predictors
        dataframe_clusters[list_of_variables_to_base_clusters] = StandardScaler().fit_transform(dataframe_clusters[list_of_variables_to_base_clusters])
    
    # Show peak-to-peak range of each predictor
    print("\nPeak-to-peak range of each predictor:")
    print(np.ptp(dataframe_clusters[list_of_variables_to_base_clusters], axis=0))

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
    model = model.fit(dataframe_clusters[list_of_variables_to_base_clusters])

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
            data=dataframe[list_of_variables_to_base_clusters + [column_name_for_clusters]],
            hue=column_name_for_clusters
        )
        plt.suptitle("Cluster Summary Plots", fontsize=15)
        plt.show()

    # Return updated dataset
    return(dataframe)

# Test the function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
iris = CreateKMeansClusters(
    dataframe=iris,
    list_of_variables_to_base_clusters=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    scale_predictor_variables=True,
    show_cluster_summary_plots=True
)

