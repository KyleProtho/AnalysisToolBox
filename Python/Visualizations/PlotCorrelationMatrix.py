# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotCorrelationMatrix(dataframe,
                          list_of_numeric_variables,
                          outcome_variable=None,
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
                          x_indent=-0.85,
                          title_y_indent=1.12,
                          subtitle_y_indent=1.09,
                          caption_y_indent=-0.35):
    # Select relevant variables, keep complete cases only
    if outcome_variable is None:
        completed_df = dataframe[list_of_numeric_variables].dropna()
    else:
        completed_df = dataframe[[outcome_variable] + list_of_numeric_variables].dropna()
    
    # Update the x_indent value if an outcome variable is specified
    if outcome_variable is None:
        x_indent = x_indent*len(list_of_numeric_variables)

    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        plt.figure(figsize=pairplot_size)
        if outcome_variable is None:
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
                x_vars=outcome_variable,
                y_vars=list_of_numeric_variables,
                kind='reg',
                markers='o',
                lowess=lowess,
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
            wrapped_x_labels = "\n".join(ax.get_xlabel()[j:j+30] for j in range(0, len(ax.get_xlabel()), 30))
            ax.set_xlabel(wrapped_x_labels, horizontalalignment='center')
            # y-axis labels
            wrapped_y_labels = "\n".join(ax.get_ylabel()[j:j+30] for j in range(0, len(ax.get_ylabel()), 30))
            ax.set_ylabel(wrapped_y_labels, horizontalalignment='center')
        
        # Format tick labels to be Arial, size 9, and color #666666
        ax.tick_params(
            which='major',
            labelsize=9,
            color='#666666'
        )
        
        # Set the title with Arial font, size 14, and color #262626 at the top of the plot
        ax.text(
            x=x_indent,
            y=title_y_indent*len(list_of_numeric_variables),
            s=title_for_plot,
            fontname="Arial",
            fontsize=14,
            color="#262626",
            transform=ax.transAxes
        )
        
        # Set the subtitle with Arial font, size 11, and color #666666
        ax.text(
            x=x_indent,
            y=subtitle_y_indent*len(list_of_numeric_variables),
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=11,
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
                if outcome_variable is None:
                    wrapped_caption = textwrap.fill(caption_for_plot, 110, break_long_words=False)
                else:
                    wrapped_caption = textwrap.fill(caption_for_plot, 60, break_long_words=False)
                
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
    
    else:
        print(completed_df.corr())


# Test the function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
PlotCorrelationMatrix(
    dataframe=iris,
    list_of_numeric_variables=[
        'sepal length (cm)',
        'sepal width (cm)', 
        'petal length (cm)', 
        'petal width (cm)'
    ],
    caption_for_plot="This is a caption for the plot. Just using this to test the word wrapping functionality."
)
# PlotCorrelationMatrix(
#     dataframe=iris, 
#     list_of_numeric_variables=[
#         'sepal width (cm)', 
#         'petal length (cm)', 
#         'petal width (cm)'
#     ],
#     outcome_variable='sepal length (cm)',
#     x_indent=-0.3,
#     caption_for_plot="This is a caption for the plot. Just using this to test the word wrapping functionality.",
# )
