import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

def PlotCorrelationMatrix(dataframe,
                          list_of_columns,
                          show_as_pairplot = True):
    # Select relevant variables, keep complete cases only
    completed_df = dataframe[list_of_columns].dropna()
    
    # Drop Inf values
    completed_df = completed_df[np.isfinite(completed_df).all(1)]
    print("Count of complete observations for correlation matrix: " + str(len(completed_df.index)))
    
    # Show pairplot if specified -- otherwise, print correlation matrix
    if show_as_pairplot:
        ax = sns.pairplot(completed_df)
        ax.fig.subplots_adjust(top=0.95)
        ax.fig.text(
            x=0.06, 
            y=1, 
            s='Correlation Pairplot', 
            fontsize=14,
            ha='left', 
            va='top'
        )
        # Show plot
        plt.show()
        
        # Clear plot
        plt.clf()
    else:
        print(completed_df.corr())

