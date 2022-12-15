import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotCorrelationMatrix(dataframe,
                          list_of_predictor_variables,
                          outcome_variable=None,
                          show_as_pairplot=True):
    # Select relevant variables, keep complete cases only
    if outcome_variable is None:
        completed_df = dataframe[list_of_predictor_variables].dropna()
    else:
        completed_df = dataframe[[outcome_variable] + list_of_predictor_variables].dropna()
    
    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        if outcome_variable is None:
            ax = sns.pairplot(
                completed_df,
                kind="reg"
            )
        else:
            ax = sns.pairplot(
                completed_df,
                kind="reg",
                x_vars=list_of_predictor_variables,
                y_vars=outcome_variable
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
    else:
        print(completed_df.corr())


# # Test the function
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# # PlotCorrelationMatrix(
# #     dataframe=iris,
# #     list_of_predictor_variables=[
# #         'sepal length (cm)',
# #         'sepal width (cm)', 
# #         'petal length (cm)', 
# #         'petal width (cm)'
# #     ]
# # )
# PlotCorrelationMatrix(
#     dataframe=iris, 
#     list_of_predictor_variables=[
#         'sepal width (cm)', 
#         'petal length (cm)', 
#         'petal width (cm)'
#     ],
#     outcome_variable='sepal length (cm)',
# )
