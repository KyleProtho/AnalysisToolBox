# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap

# Declare function
def PlotCorrelationMatrix(dataframe,
                          list_of_value_column_names,
                          list_of_outcome_column_names=None,
                          show_as_pairplot=True,
                          # Pairplot formatting options
                          pairplot_size=(20, 20),
                          scatter_fill_color="#3269a8",
                          # Line formatting options
                          fit_lowess_line=True,
                          line_color="#cc4b5a",
                          # Text formatting arguments
                          title_for_plot="Pairplot of Numeric Variables",
                          subtitle_for_plot="Shows the linear relationship between numeric variables",
                          caption_for_plot=None,
                          data_source_for_plot=None,
                          x_indent=-0.7,
                          title_y_indent=1.12,
                          title_font_size=14,
                          subtitle_y_indent=1.03,
                          subtitle_font_size=11,
                          caption_y_indent=-0.35,
                          caption_font_size=8,
                          # Plot saving arguments
                          filepath_to_save_plot=None):
    """
    Plots a correlation matrix or pairplot for a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe to plot.
        list_of_value_column_names (list): The names of the columns to include in the plot.
        list_of_outcome_column_names (list, optional): The names of the outcome columns to include in the plot. Defaults to None.
        show_as_pairplot (bool, optional): Whether to show the plot as a pairplot. Defaults to True.
        pairplot_size (tuple, optional): The size of the pairplot. Defaults to (20, 20).
        scatter_fill_color (str, optional): The fill color for scatter plots. Defaults to "#3269a8".
        fit_lowess_line (bool, optional): Whether to fit a lowess line to the data. Defaults to True.
        line_color (str, optional): The color of the line. Defaults to "#cc4b5a".
        title_for_plot (str, optional): The title for the plot. Defaults to "Pairplot of Numeric Variables".
        subtitle_for_plot (str, optional): The subtitle for the plot. Defaults to "Shows the linear relationship between numeric variables".
        caption_for_plot (str, optional): The caption for the plot. Defaults to None.
        data_source_for_plot (str, optional): The data source for the plot. Defaults to None.
        x_indent (float, optional): The x-indent for the plot. Defaults to -0.7.
        title_y_indent (float, optional): The y-indent for the title. Defaults to 1.12.
        subtitle_y_indent (float, optional): The y-indent for the subtitle. Defaults to 1.03.
        caption_y_indent (float, optional): The y-indent for the caption. Defaults to -0.35.
        filepath_to_save_plot (str, optional): The filepath to save the plot. Defaults to None.
    """
    
    # Select relevant variables, keep complete cases only
    if list_of_outcome_column_names is None:
        completed_df = dataframe[list_of_value_column_names].dropna()
    else:
        completed_df = dataframe[list_of_outcome_column_names + list_of_value_column_names].dropna()

    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        plt.figure(figsize=pairplot_size)
        if list_of_outcome_column_names is None:
            ax = sns.pairplot(
                data=completed_df,
                kind='reg',
                markers='o',
                diag_kws={
                    'color': scatter_fill_color,
                    'alpha': 0.80
                },
                plot_kws={
                    'lowess': fit_lowess_line,
                    'line_kws': {
                        'color': line_color,
                        'alpha': 0.80
                    },
                    'scatter_kws': {
                        'color': scatter_fill_color,
                        'alpha': 0.30
                    }
                }
            )
            
        else:
            ax = sns.pairplot(
                data=completed_df,
                x_vars=list_of_value_column_names,
                y_vars=list_of_outcome_column_names,
                kind='reg',
                markers='o',
                diag_kws={
                    'color': scatter_fill_color,
                    'alpha': 0.80
                },
                plot_kws={
                    'lowess': fit_lowess_line,
                    'line_kws': {
                        'color': line_color,
                        'alpha': 0.80
                    },
                    'scatter_kws': {
                        'color': scatter_fill_color,
                        'alpha': 0.30
                    }
                }
            )
            
        # Wrap labels in each subplot
        for ax in ax.axes.flat:
            # x-axis labels
            x_labels = ax.get_xlabel()
            wrapped_x_labels = "\n".join(textwrap.wrap(x_labels, 40, break_long_words=False))
            ax.set_xlabel(wrapped_x_labels, horizontalalignment='center')
            # y-axis labels
            y_labels = ax.get_ylabel()
            wrapped_y_labels = "\n".join(textwrap.wrap(y_labels, 40, break_long_words=False))
            ax.set_ylabel(wrapped_y_labels, horizontalalignment='center')
        
        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
        
        # Adjust y-indents for title and subtitle
        if list_of_outcome_column_names is None:
            title_y_indent = title_y_indent*len(list_of_value_column_names)
            subtitle_y_indent = subtitle_y_indent*len(list_of_value_column_names)
        else:
            title_y_indent = title_y_indent*len(list_of_outcome_column_names)
            subtitle_y_indent = subtitle_y_indent*len(list_of_outcome_column_names)
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent*len(list_of_value_column_names),
            y=title_y_indent,
            s=title_for_plot,
            # fontname="Arial",
            fontsize=title_font_size,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent*len(list_of_value_column_names),
            y=subtitle_y_indent,
            s=subtitle_for_plot,
            # fontname="Arial",
            fontsize=subtitle_font_size,
            color="#666666",
            transform=ax.transAxes
        )
        
        # Add a word-wrapped caption if one is provided
        if caption_for_plot != None or data_source_for_plot != None:
            # Create starting point for caption
            wrapped_caption = ""
            
            # Add the caption to the plot, if one is provided
            if caption_for_plot != None:
                # Word wrap the caption without splitting words
                if list_of_outcome_column_names is None:
                    wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
                else:
                    wrapped_caption = textwrap.fill(caption_for_plot, 60, break_long_words=False)
                
            # Add the data source to the caption, if one is provided
            if data_source_for_plot != None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            
            # Add the caption to the plot
            ax.text(
                x=x_indent*len(list_of_value_column_names),
                y=caption_y_indent,
                s=wrapped_caption,
                # fontname="Arial",
                fontsize=caption_font_size,
                color="#666666",
                transform=ax.transAxes
            )
        
        # If filepath_to_save_plot is provided, save the plot
        if filepath_to_save_plot != None:
            # Ensure that the filepath ends with '.png' or '.jpg'
            if not filepath_to_save_plot.endswith('.png') and not filepath_to_save_plot.endswith('.jpg'):
                raise ValueError("The filepath to save the plot must end with '.png' or '.jpg'.")
            
            # Save plot
            plt.savefig(
                filepath_to_save_plot, 
                bbox_inches="tight"
            )
        
        # Show plot
        plt.show()
        
        # Close plot
        plt.clf()
    
    else:
        print(completed_df.corr())

