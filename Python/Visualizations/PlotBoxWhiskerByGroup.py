import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Create box whisker function
def PlotBoxWhiskerByGroup(dataframe,
                          outcome_variable,
                          group_variable_1,
                          group_variable_2=None,
                          fill_color=None,
                          color_palette='Set1',
                          title_for_plot=None,
                          subtitle_for_plot=None):
    """_summary_
    This function generates a box whisker plot of an outcome variable by up to two grouping variables.

    Args:
        dataframe (_type_): Pandas dataframe
        outcome_variable (str): _description_
        group_variable_1 (str): _description_
        group_variable_2 (str, optional): _description_. Defaults to None.
        fill_color (str, optional): The color to fill the box whisker plot. Defaults to None. If None, the default seaborn color palette will be used.
        title_for_plot (str, optional): The title text for the plot. Defaults to None. If None, the title will be generated automatically (outcome_variable + ' by ' + group_variable_1).
        subtitle_for_plot (str, optional): The subtitle text for the plot. Defaults to None. If None, the subtitle will be generated automatically (group_variable_2).
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
    
    # Create boxplot using seaborn
    f, ax = plt.subplots(figsize=(6, 4.5))
    if group_variable_2 != None:
        if fill_color != None:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            hue=group_variable_2, 
                            color=fill_color)
        else:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            hue=group_variable_2, 
                            palette=color_palette)
    else:
        if fill_color != None:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            color=fill_color)
        else:
            ax = sns.boxplot(data=dataframe,
                            x=group_variable_1,
                            y=outcome_variable, 
                            palette=color_palette)
            
    # Remove x and y axis labels
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
    # String wrap group variable 1 tick labels
    group_variable_1_tick_labels = ax.get_xticklabels()
    group_variable_1_tick_labels = [label.get_text() for label in group_variable_1_tick_labels]
    for label in group_variable_1_tick_labels:
        label = "\n".join(label[j:j+30] for j in range(0, len(label), 30))
    ax.set_xticklabels(group_variable_1_tick_labels)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a title, with bold formatting
    plt.suptitle(
        title_for_plot,
        x=0.125,
        y=.965,
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=12,
        fontweight='bold'
    )
    
    # Add a subtitle, with normal formatting
    if group_variable_2 != None:
        plt.title(
            subtitle_for_plot,
            loc='left',
            fontsize=9
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
#     group_variable_1='species'
# )
