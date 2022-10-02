# Load packages
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Declare function
def ConductKMeansClustering(dataframe,
                            list_variables_to_form_clusters = None,
                            number_of_clusters = None,
                            standardize_variables = True,
                            random_seed = 412,
                            column_name_for_clusters = 'K-Means Cluster'):
    # Select variables specified
    if list_variables_to_form_clusters == None:
        list_variables_to_form_clusters = dataframe.columns
        df_for_clustering = dataframe.copy()
    else:
        df_for_clustering = dataframe[list_variables_to_form_clusters].copy()

    # Keep complete cases only
    df_for_clustering = df_for_clustering.dropna()

    # Standardize variables
    if standardize_variables:
        df_transformed = StandardScaler().fit_transform(df_for_clustering)
        df_transformed = pd.DataFrame(df_transformed)
        df_transformed.index = df_for_clustering.index
        df_for_clustering = df_transformed.copy()
        df_for_clustering.columns = list_variables_to_form_clusters

    # If number of clusters not specified, use ___ to find "best" number
    if number_of_clusters == None:
        model = KMeans()
        visualizer = KElbowVisualizer(
            model,
            k=(2,30),
            timings=True
        )
        visualizer.fit(df_for_clustering)
        visualizer.show()
        number_of_clusters = visualizer.elbow_value_
        
    # Conduct k-means clusters
    model = KMeans(
        n_clusters=number_of_clusters, 
        random_state=random_seed
    )
    model = model.fit(df_for_clustering)

    # Join clusters to original dataset
    df_for_clustering[column_name_for_clusters] = model.labels_

    # Convert cluster to categorical data type
    df_for_clustering[column_name_for_clusters] = df_for_clustering[column_name_for_clusters].apply(str)

    # If requested, show box plots of each cluster for each variable selected

    # Join clusters to original dataset
    df_for_clustering = df_for_clustering [[column_name_for_clusters]]
    dataframe = dataframe.merge(
        df_for_clustering,
        how='left',
        left_index=True,
        right_index=True
    )

    # Return updated dataset
    return(dataframe)
