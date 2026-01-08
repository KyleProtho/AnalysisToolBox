# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Declare function
def CreateHierarchicalClusters(dataframe,
                               list_of_value_columns_for_clustering=None,
                               number_of_clusters=None,
                               column_name_for_clusters='Hierarchical Cluster',
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
    Perform agglomerative hierarchical clustering with dendrogram visualization.

    This function applies hierarchical clustering using the agglomerative (bottom-up) approach,
    which builds a tree-like structure (dendrogram) by iteratively merging the closest clusters.
    Unlike partition-based methods, hierarchical clustering reveals the nested grouping structure
    of data at multiple levels of granularity. When the number of clusters is not specified, the
    function generates a dendrogram to help visualize the hierarchical relationships and determine
    the optimal cut point, then defaults to 3 clusters for initial analysis.

    Hierarchical clustering is essential for:
      * Taxonomy creation and classification systems
      * Gene expression analysis and biological classification
      * Document clustering and text organization
      * Social network community detection
      * Image segmentation with nested structures
      * Customer segmentation with hierarchical market structures
      * Anomaly detection through isolation in the hierarchy
      * Exploratory data analysis to understand data structure at multiple scales

    The function uses Ward's linkage method, which minimizes within-cluster variance at each
    merge step. It generates kernel density plots for each clustering variable, color-coded by
    cluster assignment, to visualize how clusters differ across features. The dendrogram provides
    a visual guide for selecting the number of clusters based on the height of merges.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to cluster. Rows with missing values in
        clustering columns will be excluded from the analysis.
    list_of_value_columns_for_clustering
        List of numeric column names to use as features for clustering. If None, all numeric
        columns in the DataFrame will be automatically selected. Defaults to None.
    number_of_clusters
        Number of clusters to create by cutting the dendrogram. If None, displays a dendrogram
        for visual inspection and defaults to 3 clusters. Defaults to None.
    column_name_for_clusters
        Name for the new column containing cluster assignments (as strings). Defaults to
        'Hierarchical Cluster'.
    random_seed
        Random seed for reproducibility (currently unused but maintained for API consistency).
        Defaults to 412.
    maximum_iterations
        Maximum iterations parameter (currently unused but maintained for API consistency).
        Defaults to 300.
    scale_clustering_column_values
        Whether to standardize features to zero mean and unit variance before clustering.
        Highly recommended for hierarchical clustering when features have different scales.
        Defaults to True.
    print_peak_to_peak_range_of_each_column
        Whether to print the peak-to-peak range of each clustering variable, useful for
        assessing scale differences before standardization. Defaults to False.
    show_cluster_summary_plots
        Whether to generate kernel density plots for each clustering variable, showing the
        distribution of values within each cluster. Defaults to True.
    color_palette
        Seaborn color palette name for cluster visualization. Options include 'Set1', 'Set2',
        'Set3', 'Pastel1', 'Dark2', etc. Defaults to 'Set2'.
    caption_for_plot
        Optional caption text to display below each density plot. Defaults to None.
    data_source_for_plot
        Optional data source attribution text, appended to caption. Defaults to None.
    show_y_axis
        Whether to display the y-axis (density scale) on summary plots. Defaults to False.
    title_y_indent
        Vertical position for plot titles relative to axes. Defaults to 1.1.
    subtitle_y_indent
        Vertical position for plot subtitles relative to axes. Defaults to 1.05.
    caption_y_indent
        Vertical position for plot captions relative to axes. Defaults to -0.15.
    summary_plot_size
        Figure size for each summary density plot as a tuple of (width, height) in inches.
        Defaults to (6, 4).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with one additional column containing cluster assignments as
        string labels (e.g., '0', '1', '2'). Rows with missing values in clustering columns
        are excluded and will have NaN in the cluster column.

    Examples
    --------
    # Customer segmentation with dendrogram-guided cluster selection
    import pandas as pd
    customer_df = pd.DataFrame({
        'customer_id': range(1, 101),
        'purchase_frequency': [5 + i % 20 for i in range(100)],
        'avg_order_value': [50 + i * 2 for i in range(100)],
        'customer_lifetime_value': [500 + i * 50 for i in range(100)]
    })
    segmented_df = CreateHierarchicalClusters(
        customer_df,
        list_of_value_columns_for_clustering=['purchase_frequency', 'avg_order_value', 'customer_lifetime_value'],
        scale_clustering_column_values=True,
        show_cluster_summary_plots=True
    )
    # Displays dendrogram, defaults to 3 clusters, shows density plots

    # Gene expression analysis with specified clusters
    gene_df = pd.DataFrame({
        'gene_id': [f'GENE_{i:04d}' for i in range(200)],
        'expression_level_1': [10 + i % 50 for i in range(200)],
        'expression_level_2': [20 + i % 40 for i in range(200)],
        'expression_level_3': [15 + i % 45 for i in range(200)]
    })
    clustered_genes = CreateHierarchicalClusters(
        gene_df,
        list_of_value_columns_for_clustering=['expression_level_1', 'expression_level_2', 'expression_level_3'],
        number_of_clusters=5,
        column_name_for_clusters='Gene Cluster',
        scale_clustering_column_values=True,
        color_palette='Set1',
        summary_plot_size=(8, 5)
    )
    # Creates 5 gene clusters without showing dendrogram

    # Document clustering with custom visualization
    document_df = pd.DataFrame({
        'doc_id': range(1, 151),
        'topic_score_1': [0.1 + i * 0.01 for i in range(150)],
        'topic_score_2': [0.2 + i * 0.005 for i in range(150)],
        'topic_score_3': [0.15 + i * 0.008 for i in range(150)],
        'readability': [50 + i % 30 for i in range(150)]
    })
    doc_clusters = CreateHierarchicalClusters(
        document_df,
        list_of_value_columns_for_clustering=['topic_score_1', 'topic_score_2', 'topic_score_3', 'readability'],
        number_of_clusters=4,
        scale_clustering_column_values=True,
        show_cluster_summary_plots=True,
        caption_for_plot='Hierarchical clustering reveals nested document categories',
        data_source_for_plot='Internal Document Repository',
        show_y_axis=True
    )
    # Creates 4 document clusters with custom captions and y-axis displayed

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
    
    # If number of clusters not specified, create a dendrogram to help determine it
    if number_of_clusters == None:
        print("\n\nReview the dendrogram and determine the optimal number of clusters.")
        print("Hierarchical clustering will proceed with 3 clusters, but you can change this value by setting the number_of_clusters argument, if needed.")
        linked = linkage(dataframe_clusters[list_of_value_columns_for_clustering], 'ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.show()
        # Set clusters to 3
        number_of_clusters = 3

    # Conduct hierarchical clustering
    model = AgglomerativeClustering(n_clusters=number_of_clusters)
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
    
    # Show cluster summary plots if requested
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

