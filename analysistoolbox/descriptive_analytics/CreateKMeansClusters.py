# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    Perform K-Means clustering with automatic elbow method optimization.

    This function applies K-Means clustering, a centroid-based partitioning algorithm that
    groups data into K distinct, non-overlapping clusters by minimizing within-cluster variance.
    The algorithm iteratively assigns observations to the nearest cluster centroid and updates
    centroids until convergence. When the number of clusters is not specified, the function
    uses the elbow method with distortion and timing metrics to automatically identify the
    optimal K value, balancing cluster cohesion with computational efficiency.

    K-Means clustering is essential for:
      * Customer segmentation and behavioral grouping
      * Market basket analysis and product categorization
      * Image compression and color quantization
      * Document clustering and topic grouping
      * Anomaly detection through distance from centroids
      * Data preprocessing for supervised learning
      * Geographic clustering and location-based analysis
      * Performance optimization through data summarization

    The function uses the elbow method to plot within-cluster sum of squares (distortion) against
    the number of clusters, identifying the "elbow point" where adding more clusters yields
    diminishing returns. It generates kernel density plots for each clustering variable, showing
    how cluster distributions differ across features. Feature scaling is highly recommended and
    enabled by default to ensure equal weighting of variables.

    Parameters
    ----------
    dataframe
        A pandas DataFrame containing the data to cluster. Rows with missing values in
        clustering columns will be excluded from the analysis.
    list_of_value_columns_for_clustering
        List of numeric column names to use as features for clustering. If None, all numeric
        columns in the DataFrame will be automatically selected. Defaults to None.
    number_of_clusters
        Number of clusters (K) to create. If None, uses the elbow method to automatically
        determine the optimal number between 2 and 20 (or number of observations, whichever
        is smaller). Defaults to None.
    column_name_for_clusters
        Name for the new column containing cluster assignments (as strings). Defaults to
        'K-Means Cluster'.
    random_seed
        Random seed for reproducibility of centroid initialization. K-Means uses random
        initialization, so setting this ensures consistent results. Defaults to 412.
    maximum_iterations
        Maximum number of iterations for the K-Means algorithm to converge. Increase if
        convergence warnings appear. Defaults to 300.
    scale_clustering_column_values
        Whether to standardize features to zero mean and unit variance before clustering.
        Highly recommended for K-Means when features have different scales. Defaults to True.
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
    # Customer segmentation with automatic elbow method
    import pandas as pd
    customer_df = pd.DataFrame({
        'customer_id': range(1, 301),
        'annual_revenue': [1000 + i * 100 for i in range(300)],
        'purchase_frequency': [2 + i % 50 for i in range(300)],
        'avg_transaction_value': [50 + i % 200 for i in range(300)]
    })
    segmented_df = CreateKMeansClusters(
        customer_df,
        list_of_value_columns_for_clustering=['annual_revenue', 'purchase_frequency', 'avg_transaction_value'],
        scale_clustering_column_values=True,
        show_cluster_summary_plots=True
    )
    # Automatically determines optimal K using elbow method, shows density plots

    # Product categorization with specified clusters
    product_df = pd.DataFrame({
        'product_id': [f'PROD_{i:04d}' for i in range(200)],
        'price': [10 + i % 100 for i in range(200)],
        'sales_volume': [100 + i * 5 for i in range(200)],
        'profit_margin': [0.1 + (i % 30) / 100 for i in range(200)],
        'customer_rating': [3.0 + (i % 20) / 10 for i in range(200)]
    })
    categorized_df = CreateKMeansClusters(
        product_df,
        list_of_value_columns_for_clustering=['price', 'sales_volume', 'profit_margin', 'customer_rating'],
        number_of_clusters=5,
        column_name_for_clusters='Product Category',
        scale_clustering_column_values=True,
        color_palette='husl',
        random_seed=42
    )
    # Creates 5 product categories without elbow method

    # Geographic clustering with custom visualization
    location_df = pd.DataFrame({
        'store_id': range(1, 151),
        'latitude': [40.7 + i * 0.01 for i in range(150)],
        'longitude': [-74.0 + i * 0.01 for i in range(150)],
        'daily_foot_traffic': [500 + i * 10 for i in range(150)],
        'avg_sale_per_customer': [25 + i % 50 for i in range(150)]
    })
    clustered_stores = CreateKMeansClusters(
        location_df,
        list_of_value_columns_for_clustering=['latitude', 'longitude', 'daily_foot_traffic', 'avg_sale_per_customer'],
        number_of_clusters=6,
        column_name_for_clusters='Store Cluster',
        scale_clustering_column_values=True,
        show_cluster_summary_plots=True,
        caption_for_plot='K-Means clustering identifies geographic store segments',
        data_source_for_plot='Retail Analytics Database',
        show_y_axis=True,
        summary_plot_size=(8, 5)
    )
    # Creates 6 geographic clusters with custom captions and larger plots

    """
    # Lazy load uncommon packages
    from yellowbrick.cluster import KElbowVisualizer
    
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

