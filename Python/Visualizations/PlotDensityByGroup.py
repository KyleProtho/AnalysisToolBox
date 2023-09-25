# Load packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def PlotDensityByGroup(dataframe,
                       numeric_variable,
                       grouping_variable,
                       color_palette='Set2',
                       # Text formatting arguments
                       title_for_plot=None,
                       subtitle_for_plot=None,
                       caption_for_plot=None,
                       data_source_for_plot=None,
                       show_y_axis=False,
                       title_y_indent=1.1,
                       subtitle_y_indent=1.05,
                       caption_y_indent=-0.15,
                       # Plot formatting arguments
                       figure_size=(8, 6)):
    """_summary_
    This function generates a density plot of a numeric variable by a grouping variable.

    Args:
        dataframe (_type_): Pandas dataframe
        numeric_variable (str): The column name of the numeric variable to plot.
        grouping_variable (str): The column name of the grouping variable.
        color_palette (str, optional): The seaborn color palette to use. Defaults to 'Set2'.
        title_for_plot (str, optional): The title text for the plot. Defaults to None. If None, the title will be generated automatically (numeric_variable + ' by ' + grouping_variable).
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
        
    # Generate density plot using seaborn
    sns.kdeplot(
        data=dataframe,
        x=numeric_variable,
        hue=grouping_variable,
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


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# PlotDensityByGroup(
#     dataframe=iris, 
#     numeric_variable='sepal length (cm)', 
#     grouping_variable='species',
#     title_for_plot='Sepal Length by Species',
#     subtitle_for_plot="Shows the distribution of sepal length by species.",
#     data_source_for_plot='Iris Data Set',
# )
