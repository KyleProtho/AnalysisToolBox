# Load packages
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def PlotScatterplot(dataframe,
                    y_axis_variable,
                    x_axis_variable,
                    dot_fill_color="#999999",
                    grouping_variable=None,
                    fitted_line_type=None,
                    group_color_palette="Set1",
                    title_for_plot=None,
                    subtitle_for_plot=None,
                    caption_for_plot=None,
                    data_source_for_plot=None,
                    line_color=None,
                    x_indent=-0.128,
                    title_y_indent=1.125,
                    subtitle_y_indent=1.05,
                    caption_y_indent=-0.3,
                    folder_to_save_plot=None):
    
    # Ensure that fitted_line_type is None, 'straight', or 'lowess'
    if fitted_line_type not in [None, 'straight', 'lowess']:
        raise ValueError("Invalied fitted_line_type argument. Please enter None, 'straight', or 'lowess'.")
    
    # If line color is not specified, use fill color
    if line_color == None:
        line_color = dot_fill_color
    
    # Draw scatterplot
    if fitted_line_type == None:
        if grouping_variable == None:
            ax = sns.scatterplot(
                data=dataframe, 
                x=x_axis_variable,
                y=y_axis_variable,
                color=dot_fill_color,
                alpha=0.5,
                linewidth=0.5,
                edgecolor=dot_fill_color
            )
        else:
            ax = sns.scatterplot(
                data=dataframe, 
                x=x_axis_variable,
                y=y_axis_variable,
                hue=grouping_variable,
                alpha=0.5,
                linewidth=0.5,
                palette=group_color_palette
            )
    elif fitted_line_type == 'straight':
        if grouping_variable == None:
            ax = sns.regplot(
                data=dataframe,
                x=x_axis_variable,
                y=y_axis_variable,
                color=dot_fill_color,
                alpha=0.5,
                linewidth=0.5,
                edgecolor=dot_fill_color,
                fit_reg=True,
                line_kws={'color': line_color}
            )
        else:
            ax = sns.lmplot(
                data=dataframe,
                x=x_axis_variable,
                y=y_axis_variable,
                hue=grouping_variable,
                palette=group_color_palette,
                fit_reg=True
            )
    elif fitted_line_type == 'lowess':
        if grouping_variable == None:
            ax = sns.regplot(
                data=dataframe,
                x=x_axis_variable,
                y=y_axis_variable,
                marker='o',
                scatter_kws={'color': dot_fill_color,
                                'alpha': 0.5,
                                'linewidth': 0.5,
                                'edgecolor': dot_fill_color},
                lowess=True,
                line_kws={'color': line_color}
            )
        else:
            ax = sns.lmplot(
                data=dataframe,
                x=x_axis_variable,
                y=y_axis_variable,
                hue=grouping_variable,
                palette=group_color_palette,
                scatter_kws={'alpha': 0.5,
                             'linewidth': 0.5},
                lowess=True
            )
        
    # Remove top and right spines, and set bottom and left spines to gray
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
    except AttributeError:
        ax.ax.spines['top'].set_visible(False)
        ax.ax.spines['right'].set_visible(False)
        ax.ax.spines['bottom'].set_color('#666666')
        ax.ax.spines['left'].set_color('#666666')
    
    # Format tick labels to be Arial, size 9, and color #666666
    try:
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
    except AttributeError:
        ax.ax.tick_params(
            which='major', 
            labelsize=9, 
            color='#666666'
        )
    
    # Set the title with Arial font, size 14, and color #262626 at the top of the plot
    try:
        ax.text(
            x=x_indent,
            y=title_y_indent,
            s=title_for_plot,
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
    except AttributeError:
        ax.ax.text(
            x=x_indent,
            y=title_y_indent-0.05,
            s=title_for_plot,
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.ax.transAxes
        )
        
    # Set the subtitle with Arial font, size 11, and color #666666
    try:
        ax.text(
            x=x_indent,
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.transAxes
        )
    except AttributeError:
        ax.ax.text(
            x=x_indent,
            y=subtitle_y_indent-0.023,
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=11,
            color="#666666",
            transform=ax.ax.transAxes
        )
        
    # Move the y-axis label to the top of the y-axis, and set the font to Arial, size 9, and color #666666
    try:
        ax.yaxis.set_label_coords(-0.1, 0.84)
        ax.yaxis.set_label_text(
            y_axis_variable,
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
    except AttributeError:
        ax.ax.yaxis.set_label_coords(-0.1, 0.87)
        ax.ax.yaxis.set_label_text(
            y_axis_variable,
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
    # Move the x-axis label to the right of the x-axis, and set the font to Arial, size 9, and color #666666
    try:
        ax.xaxis.set_label_coords(0.9, -0.1)
        ax.xaxis.set_label_text(
            x_axis_variable,
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
    except AttributeError:
        ax.ax.xaxis.set_label_coords(0.9, -0.075)
        ax.ax.xaxis.set_label_text(
            x_axis_variable,
            fontname="Arial",
            fontsize=10,
            color="#666666"
        )
        
    # Add a word-wrapped caption if one is provided
    if caption_for_plot != None or data_source_for_plot != None:
        if caption_for_plot != None:
            # Word wrap the caption without splitting words
            if len(caption_for_plot) > 120:
                # Split the caption into words
                words = caption_for_plot.split(" ")
                # Initialize the wrapped caption
                wrapped_caption = ""
                # Initialize the line length
                line_length = 0
                # Iterate through the words
                for word in words:
                    # If the word is too long to fit on the current line, add a new line
                    if line_length + len(word) > 120:
                        wrapped_caption = wrapped_caption + "\n"
                        line_length = 0
                    # Add the word to the line
                    wrapped_caption = wrapped_caption + word + " "
                    # Update the line length
                    line_length = line_length + len(word) + 1
        else:
            wrapped_caption = ""
            
        # Add the data source to the caption, if one is provided
        if data_source_for_plot != None:
            wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
        
        # Add the caption to the plot
        try:
            ax.text(
                x=x_indent,
                y=caption_y_indent,
                s=wrapped_caption,
                fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.transAxes
            )
        except AttributeError:
            ax.ax.text(
                x=x_indent,
                y=caption_y_indent+0.075,
                s=wrapped_caption,
                fontname="Arial",
                fontsize=8,
                color="#666666",
                transform=ax.ax.transAxes
            )
        
    # Save plot to folder if one is specified
    if folder_to_save_plot != None:
        try:
            plot_filepath = str(folder_to_save_plot) + "Scatterplot - " + categorical_variable + ".png"
            plot_filepath = os.path.normpath(plot_filepath)
            fig.savefig(plot_filepath)
        except:
            print("The filepath you entered to save your plot is invalid. Plot not saved.")
    
    # Show plot
    plt.show()
    
    # Clear plot
    plt.clf()


# # Test function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# iris['species'] = iris['species'].astype('category')
# PlotScatterplot(
#     dataframe=iris,
#     y_axis_variable="sepal length (cm)",
#     x_axis_variable="petal length (cm)",
#     title_for_plot="Sepal Length vs. Petal Length",
#     subtitle_for_plot="Generally, sepal length increases as petal length increases.",
#     caption_for_plot="This is a caption for the plot. It can be up to 120 characters long, and it will be word-wrapped without splitting words.",
#     data_source_for_plot="https://archive.ics.uci.edu/ml/datasets/iris"
# )
# PlotScatterplot(
#     dataframe=iris,
#     y_axis_variable="sepal length (cm)",
#     x_axis_variable="petal length (cm)",
#     title_for_plot="Sepal Length vs. Petal Length",
#     subtitle_for_plot="Generally, sepal length increases as petal length increases.",
#     caption_for_plot="This is a caption for the plot. It can be up to 120 characters long, and it will be word-wrapped without splitting words.",
#     data_source_for_plot="https://archive.ics.uci.edu/ml/datasets/iris",
#     fitted_line_type="lowess",
#     line_color="#d14a41",
# )
# PlotScatterplot(
#     dataframe=iris,
#     y_axis_variable="sepal length (cm)",
#     x_axis_variable="petal length (cm)",
#     title_for_plot="Sepal Length vs. Petal Length",
#     subtitle_for_plot="Generally, sepal length increases as petal length increases.",
#     caption_for_plot="This is a caption for the plot. It can be up to 120 characters long, and it will be word-wrapped without splitting words.",
#     data_source_for_plot="https://archive.ics.uci.edu/ml/datasets/iris",
#     grouping_variable="species",
# )
# PlotScatterplot(
#     dataframe=iris,
#     y_axis_variable="sepal length (cm)",
#     x_axis_variable="petal length (cm)",
#     title_for_plot="Sepal Length vs. Petal Length",
#     subtitle_for_plot="Generally, sepal length increases as petal length increases.",
#     caption_for_plot="This is a caption for the plot. It can be up to 120 characters long, and it will be word-wrapped without splitting words.",
#     data_source_for_plot="https://archive.ics.uci.edu/ml/datasets/iris",
#     fitted_line_type="lowess",
#     grouping_variable="species",
# )
