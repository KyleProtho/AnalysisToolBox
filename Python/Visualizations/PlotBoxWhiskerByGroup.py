import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

# Create box whisker function
def PlotBoxWhiskerByGroup(dataframe,
                          outcome_variable,
                          group_variable_1,
                          group_variable_2=None,
                          fill_color=None,
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
                          x_indent=-0.128,
                          # Plot formatting arguments
                          figure_size=(8, 6)):
    """
    Function to create a box and whisker plot for a given outcome variable, grouped by one or two categorical variables. 

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    outcome_variable (str): The name of the outcome variable.
    group_variable_1 (str): The name of the first group variable.
    group_variable_2 (str, optional): The name of the second group variable. Defaults to None.
    fill_color (str, optional): The color to fill the box plot with. Defaults to None.
    color_palette (str, optional): The color palette to use for the box plot. Defaults to 'Set2'.
    title_for_plot (str, optional): The title for the plot. Defaults to None.
    subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to None.
    caption_for_plot (str, optional): The caption for the plot. Defaults to None.
    data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
    show_y_axis (bool, optional): Whether to show the y-axis. Defaults to False.
    title_y_indent (float, optional): The y-indent for the title. Defaults to 1.1.
    subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.05.
    caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.15.
    x_indent (float, optional): The x-indent for the plot. Defaults to -0.128.
    figure_size (tuple, optional): The size of the plot. Defaults to (8, 6).

    Returns:
    None
    """
    
    # If no plot title is specified, generate one
    if title_for_plot == None:
        if group_variable_2 == None:
            title_for_plot = outcome_variable + ' by ' + group_variable_1
        else:
            title_for_plot = outcome_variable
            
    # If no plot subtitle is specified, generate one
    if subtitle_for_plot == None and group_variable_2 != None:
        subtitle_for_plot = ' by ' + group_variable_1 + ' and ' + group_variable_2
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Generate box whisker plot
    if group_variable_2 != None:
        if fill_color != None:
            ax = sns.boxplot(
                data=dataframe,
                x=group_variable_1,
                y=outcome_variable, 
                hue=group_variable_2, 
                color=fill_color
            )
        else:
            ax = sns.boxplot(
                data=dataframe,
                x=group_variable_1,
                y=outcome_variable, 
                hue=group_variable_2, 
                palette=color_palette
            )
    else:
        if fill_color != None:
            ax = sns.boxplot(
                data=dataframe,
                x=group_variable_1,
                y=outcome_variable, 
                color=fill_color
            )
        else:
            ax = sns.boxplot(
                data=dataframe,
                x=group_variable_1,
                y=outcome_variable, 
                palette=color_palette
            )
    
    # Remove top, and right spines. Set bottom and left spine to dark gray.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#262626")
    ax.spines['left'].set_color("#262626")
    
    # Add space between the title and the plot
    plt.subplots_adjust(top=0.85)
    
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
    
    # String wrap group variable 1 tick labels
    group_variable_1_tick_labels = ax.get_xticklabels()
    group_variable_1_tick_labels = [label.get_text() for label in group_variable_1_tick_labels]
    for label in group_variable_1_tick_labels:
        label = textwrap.fill(label, 30, break_long_words=False)
    ax.set_xticklabels(group_variable_1_tick_labels)
    
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
    
    # Set y-axis tick label font to Arial, size 9, and color #666666
    ax.tick_params(
        axis='y',
        which='major',
        labelsize=9,
        labelcolor="#666666",
        pad=2,
        bottom=True,
        labelbottom=True
    )
    plt.yticks(fontname='Arial')
    
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


# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# PlotBoxWhiskerByGroup(
#     dataframe=iris,
#     outcome_variable='sepal length (cm)',
#     group_variable_1='species',
#     title_for_plot='Sepal Length by Species',
#     subtitle_for_plot='A test visulaization using the iris dataset',
# )
