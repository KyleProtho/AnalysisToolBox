# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def TwoSampleTTestOfIndependence(dataframe,
                                 outcome_column,
                                 grouping_column,
                                 confidence_interval=.95,
                                 alternative_hypothesis="two-sided",
                                 homogeneity_of_variance=True,
                                 null_handling='omit',
                                 plot_sample_distributions=True,
                                 # Plotting arguments
                                 color_palette='Set2',
                                 title_for_plot="T-Test of Independence",
                                 subtitle_for_plot="Shows the distribution of the sample means for each group.",
                                 caption_for_plot=None,
                                 data_source_for_plot=None,
                                 show_y_axis=False,
                                 title_y_indent=1.1,
                                 subtitle_y_indent=1.05,
                                 caption_y_indent=-0.15,
                                 figure_size=(8, 6)):
    # Get groupings (there should only be two)
    list_groups = dataframe[grouping_column].unique()
    if len(list_groups) > 2:
        raise ValueError("T-Test can only be performed with two groups. The grouping column you selected has " + str(len(list_groups)) + " groups.")
    
    # Show pivot table summary
    table_groups = pd.pivot_table(dataframe, 
        values=[outcome_column], 
        index=[grouping_column],
        aggfunc={outcome_column: [len, min, max, np.mean, np.std]}
    )
    print(table_groups)
    print("\n")
    
    # Restructure data
    dataframe = dataframe[[
        outcome_column,
        grouping_column,
    ]]
    group1 = dataframe[dataframe[grouping_column]==list_groups[0]]
    group2 = dataframe[dataframe[grouping_column]==list_groups[1]]
    
    # Conduct two sample t-test
    ttest_results = ttest_ind(
        group1[outcome_column], 
        group2[outcome_column], 
        axis=0,
        nan_policy=null_handling, 
        alternative=alternative_hypothesis,
        equal_var=homogeneity_of_variance
    )
    
    # Get the p-value
    p_value = ttest_results.pvalue
    
    # Get the test statistic
    test_statistic = ttest_results.statistic
    
    # If plot_distribution_of_differences is True, plot distribution of differences
    if plot_sample_distributions:
        # Create placeholder for results
        data_distribution = pd.DataFrame()
        
        # Get unqiue groups
        list_of_groups = dataframe[grouping_column].unique()
        
        # Iterate through groups
        for group in list_of_groups:
            # Get the group's data
            group_data = dataframe[dataframe[grouping_column] == group][outcome_column]
            
            # Get the group's mean
            group_mean = group_data.mean()
            
            # Get the group's standard error
            group_se = group_data.sem()
            
            # Generate a normal distribution
            group_distribution = np.random.normal(group_mean, group_se, 10000)
            
            # Convert to dataframe
            df_group_distribution = pd.DataFrame(group_distribution,
                                                 columns=[outcome_column])
            
            # Add group name
            df_group_distribution[grouping_column] = group
            
            # Row bind to data_distribution
            data_distribution = pd.concat([data_distribution, df_group_distribution], axis=0).reset_index(drop=True)
            
        # Plot distribution of means
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figure_size)
            
        # Generate density plot using seaborn
        sns.kdeplot(
            data=data_distribution,
            x=outcome_column,
            hue=grouping_column,
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
            s=title_for_plot,
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
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

    # Print interpretation of results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of the groups.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of the " + str(list_groups[0]) + " group is " + alternative_hypothesis + " than that of the " + str(list_groups[1]) + " group to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(p_value, 3)))
    print("statistic:", str(round(test_statistic, 3)))
    
    # Return results
    return ttest_results


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# iris['species'] = iris['species'].astype('category')
# # Filter to two groups
# iris = iris[iris['species'].isin([0, 1])]
# # Run the function
# TwoSampleTTestOfIndependence(
#     dataframe=iris,
#     outcome_column="sepal length (cm)",
#     grouping_column="species",
#     confidence_interval=.95,
#     alternative_hypothesis="two-sided",
# )
