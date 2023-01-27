import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotDensityByGroup(dataframe,
                       numeric_variable,
                       grouping_variable,
                       color_palette='Set2',
                       title_for_plot=None):
    """_summary_
    This function generates a density plot of a numeric variable by a grouping variable.

    Args:
        dataframe (_type_): Pandas dataframe
        numeric_variable (str): The column name of the numeric variable to plot.
        grouping_variable (str): The column name of the grouping variable.
        color_palette (str, optional): The seaborn color palette to use. Defaults to 'Set2'.
        title_for_plot (str, optional): The title text for the plot. Defaults to None. If None, the title will be generated automatically (numeric_variable + ' by ' + grouping_variable).
    """
    
    # Set plot title if not specified
    if title_for_plot == None:
        title_for_plot = numeric_variable + ' by ' + grouping_variable
        
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
    
    # Remove top and right borders
    sns.despine()
    
    # Add a title, with bold formatting
    plt.title(
        title_for_plot,
        size=12,
        fontweight='bold',
        loc='left'
    )
    
    # Show plot
    plt.show()


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# iris['species'] = datasets.load_iris(as_frame=True).target
# PlotDensityByGroup(
#     dataframe=iris, 
#     numeric_variable='sepal length (cm)', 
#     grouping_variable='species'
# )
