import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotCorrelationMatrix(dataframe,
                          list_of_numeric_variables,
                          outcome_variable=None,
                          show_as_pairplot=True,
                          scatter_fill_color="#3269a8",
                          line_color="#cc4b5a"):
    # Select relevant variables, keep complete cases only
    if outcome_variable is None:
        completed_df = dataframe[list_of_numeric_variables].dropna()
    else:
        completed_df = dataframe[[outcome_variable] + list_of_numeric_variables].dropna()
    
    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        if outcome_variable is None:
            ax = sns.PairGrid(completed_df)
            ax.map_lower(
                sns.regplot,
                lowess=True,
                scatter_kws={'alpha':0.30}, 
                line_kws={'color': line_color}
            )
        else:
            ax = sns.PairGrid(
                completed_df,
                x_vars=list_of_numeric_variables,
                y_vars=outcome_variable
            )
            ax.map(
                sns.regplot,
                lowess=True,
                scatter_kws={'alpha':0.30}, 
                line_kws={'color': line_color}
            )
        ax.fig.subplots_adjust(top=0.95)
        ax.fig.text(
            x=0.06, 
            y=1, 
            s='Correlation Pairplot', 
            fontsize=14,
            ha='left', 
            va='top'
        )
        # Wrap labels in each subplot
        for ax in ax.axes.flat:
            # x-axis labels
            wrapped_x_labels = "\n".join(ax.get_xlabel()[j:j+30] for j in range(0, len(ax.get_xlabel()), 30))
            ax.set_xlabel(wrapped_x_labels, horizontalalignment='center')
            # y-axis labels
            wrapped_y_labels = "\n".join(ax.get_ylabel()[j:j+30] for j in range(0, len(ax.get_ylabel()), 30))
            ax.set_ylabel(wrapped_y_labels, horizontalalignment='center')
    else:
        print(completed_df.corr())


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# # PlotCorrelationMatrix(
# #     dataframe=iris,
# #     list_of_numeric_variables=[
# #         'sepal length (cm)',
# #         'sepal width (cm)', 
# #         'petal length (cm)', 
# #         'petal width (cm)'
# #     ]
# # )
# PlotCorrelationMatrix(
#     dataframe=iris, 
#     list_of_numeric_variables=[
#         'sepal width (cm)', 
#         'petal length (cm)', 
#         'petal width (cm)'
#     ],
#     outcome_variable='sepal length (cm)',
# )
