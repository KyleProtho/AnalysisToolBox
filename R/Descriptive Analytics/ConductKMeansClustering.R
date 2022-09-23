# Load packages
library(dplyr)
library(ggplot2)
library(mclust)

# Declare function
ConductKMeansClustering = function(dataframe,
                                   list_of_variables_to_create_clusters,
                                   standardize_variables = TRUE,
                                   number_of_clusters = NULL,
                                   column_name_for_clusters = "kmeans_cluster",
                                   show_cluster_modeling_results = TRUE,
                                   random_seed = 412) {
  # Set seed, if specified
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  
  # Select variables specified, and keep complete cases only
  dataframe_clust = dataframe %>% select(
    one_of(list_of_variables_to_create_clusters)
  ) %>% na.omit
  
  # Standardize variables
  if (standardize_variables) {
    dataframe_clust = scale(dataframe_clust)
    dataframe_clust = as.data.frame(dataframe_clust)
  }
  
  # If number_of_clusters not specified, use mclust to identify the most likely model and number of clusters
  if (is.null(number_of_clusters)) {
    cluster_model = Mclust(dataframe_clust)
    number_of_clusters = cluster_model$G
    if (show_cluster_modeling_results) {
      print(cluster_model$BIC)
    }
  }
  
  # Conduct k-means clustering
  cluster_results = kmeans(dataframe_clust, number_of_clusters)
  
  # Bind clusters to original dataframe
  df_clusters = as.data.frame(cluster_results$cluster)
  dataframe = bind_cols(
    dataframe,
    df_clusters
  ) 
  names(dataframe)[names(dataframe) == "cluster_results$cluster"] = column_name_for_clusters
  
  # Convert to cluster to factor type
  dataframe[[column_name_for_clusters]] = as.factor(dataframe[[column_name_for_clusters]])
  
  # Import box-whisker function from SnippetsForStatistics
  source("https://raw.githubusercontent.com/onenonlykpro/SnippetsForStatistics/master/R/Visualizations/PlotBoxWhiskerByGroup.R")
  
  # Visualize variables within each cluster using boxplots
  for (variable in list_of_variables_to_create_clusters) {
    PlotBoxWhiskerByGroup(dataframe = dataframe,
                          quantitative_variable = variable,
                          grouping_variable_1 = column_name_for_clusters)
  }
  
  # Return dataframe
  return(dataframe)
}
