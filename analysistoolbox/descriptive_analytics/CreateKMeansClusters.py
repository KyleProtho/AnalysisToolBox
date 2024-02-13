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
                         list_of_value_columns_for_clustering=None,
                         number_of_clusters=None,
                         column_name_for_clusters='K-Means Cluster',
                         random_seed=412,
                         maximum_iterations=300,
                         scale_clustering_column_values=True,
                         # Output arguments
                         print_peak_to_peak_range_of_each_column=False,
                         show_cluster_summary_plots=True,
                         color_palette='Set2',
                         # Text formatting arguments
                         caption_for_plot=None,
                         data_source_for_plot=None,
                         show_y_axis=False,
                         title_y_indent=1.1,
                         subtitle_y_indent=1.05,
                         caption_y_indent=-0.15,
                         # Plot formatting arguments
                         summary_plot_size=(6, 4)):
    """
    This function creates K-Means clusters on a dataset based on the variables specified.

    Args:
        dataframe (Pandas dataframe): Pandas dataframe containing the data to be analyzed.
        list_of_value_columns_for_clustering (list, optional): The list of variables to base the clusters on. Defaults to None, which will use all variables in the dataframe.
        number_of_clusters (int, optional): The number of clusters to create. Defaults to None, which will use the elbow method to determine the optimal number of clusters.
        column_name_for_clusters (str, optional): The name of the new column containing the clusters. Defaults to 'K-Means Cluster'.
        scale_clustering_column_values (bool, optional): Whether to scale the predictor variables prior to analysis. Defaults to True.
        show_cluster_summary_plots (bool, optional): Whether to show cluster summary plots. Defaults to True.
        summary_plot_size (tuple, optional): The size of the summary plots. Defaults to (20, 20).
        random_seed (int, optional): The random seed to use for replication. Defaults to 412.
        maximum_iterations (int, optional): The maximum number of iterations to use for the K-Means algorithm. Defaults to 300.
    
    Returns:
        Pandas dataframe: An updated Pandas dataframe with the clusters joined to the original data.
    """
    # If no list of variables specified, use all numeric variables
    if list_of_value_columns_for_clustering == None:
        list_of_value_columns_for_clustering = list(dataframe.select_dtypes(include=np.number).columns)
    
    # Keep complete cases only
    dataframe_clusters = dataframe.dropna(subset=list_of_value_columns_for_clustering)

    # Scale the predictors, if requested
    if scale_clustering_column_values:
        # Scale predictors
        dataframe_clusters[list_of_value_columns_for_clustering] = StandardScaler().fit_transform(dataframe_clusters[list_of_value_columns_for_clustering])
    
    # Show peak-to-peak range of each variable
    if print_peak_to_peak_range_of_each_column:
        print("\nPeak-to-peak range of each value column/variable:")
        print(np.ptp(dataframe_clusters[list_of_value_columns_for_clustering], axis=0))
    
    # If number of clusters not specified, use elbow to find "best" number
    if number_of_clusters == None:
        # Maximum number of clusters is 20 or the number of observations, whichever is smaller
        max_clusters = min(20, dataframe_clusters.shape[0])
        
        # Conduct elbow method
        model = KMeans()
        visualizer = KElbowVisualizer(
            model,
            k=(2, max_clusters),
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
    model = model.fit(dataframe_clusters[list_of_value_columns_for_clustering])

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
    
    if show_cluster_summary_plots:
        # Loop through each numeric variable
        for numeric_var in list_of_value_columns_for_clustering:
            # Create a temporary dataframe for the cluster
            dataframe_temp = dataframe[[column_name_for_clusters, numeric_var]]
            
            # Rename the variable to 'Value'
            dataframe_temp = dataframe_temp.rename(
                columns={numeric_var: 'Value'}
            )
            
            # Add the variable name to the dataframe
            dataframe_temp['Variable'] = numeric_var
            
            # Reorder the columns
            dataframe_temp = dataframe_temp[[column_name_for_clusters, 'Variable', 'Value']]
            
            # Create figure and axes
            fig, ax = plt.subplots(figsize=summary_plot_size)
                
            # Generate density plot using seaborn
            sns.kdeplot(
                data=dataframe_temp,
                x='Value',
                hue=column_name_for_clusters,
                fill=True,
                common_norm=False,
                alpha=.4,
                palette=color_palette,
                linewidth=1
            )
            
            # Remove top, left, and right spines. Set bottom spine to dark gray.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color("#262626")
            
            # Remove the y-axis, and adjust the indent of the plot titles
            if show_y_axis == False:
                ax.axes.get_yaxis().set_visible(False)
                x_indent = 0.015
            else:
                x_indent = -0.005
            
            # Set the title with Arial font, size 14, and color #262626 at the top of the plot
            ax.text(
                x=x_indent,
                y=title_y_indent,
                s="Cluster Summary Plot",
                fontname="Arial",
                fontsize=14,
                color="#262626",
                transform=ax.transAxes
            )
            
            # Set the subtitle with Arial font, size 11, and color #666666
            ax.text(
                x=x_indent,
                y=subtitle_y_indent,
                s="Variable: " + numeric_var,
                fontname="Arial",
                fontsize=11,
                color="#666666",
                transform=ax.transAxes
            )
            
            # Set x-axis tick label font to Arial, size 9, and color #666666
            ax.tick_params(
                axis='x',
                which='major',
                labelsize=9,
                labelcolor="#666666",
                pad=2,
                bottom=True,
                labelbottom=True
            )
            plt.xticks(fontname='Arial')
            
            # Add a word-wrapped caption if one is provided
            if caption_for_plot != None or data_source_for_plot != None:
                # Create starting point for caption
                wrapped_caption = ""
                
                # Add the caption to the plot, if one is provided
                if caption_for_plot != None:
                    # Word wrap the caption without splitting words
                    wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
                    
                # Add the data source to the caption, if one is provided
                if data_source_for_plot != None:
                    wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
                
                # Add the caption to the plot
                ax.text(
                    x=x_indent,
                    y=caption_y_indent,
                    s=wrapped_caption,
                    fontname="Arial",
                    fontsize=8,
                    color="#666666",
                    transform=ax.transAxes
                )
                
            # Show plot
            plt.show()
            
            # Clear plot
            plt.clf()

    # Return updated dataset
    return(dataframe)

