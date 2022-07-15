# Load packages
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import seaborn as sns
sns.set(style="white",
        font="Arial",
        context="paper")

# Declare function
def ChiSquareTestOfIndependence(dataframe,
                                categorical_outcome_variable,
                                categorical_predictor_variable,
                                show_contingency_tables = True,
                                show_plot = True):
    # Select necessary columns
    dataframe = dataframe[[
        categorical_predictor_variable,
        categorical_outcome_variable
    ]]
    
    # Create contingency table in Python
    contingency_table = sm.stats.Table.from_data(dataframe)
    
    # Perform chi-square test of independence
    results = chi2_contingency(contingency_table.table_orig)
    
    # Print results of test
    print("Chi-square test statistic: " + str(round(results[0], 3)))
    print("p-value: " + str(round(results[1], 3)))
    print("Degrees of freedom: " + str(round(results[2], 3)))
    
    # Show contingency table with margins
    if show_contingency_tables:
        print("")
        print("Observed:")
        print(contingency_table.table_orig)
        print("")
        print("Expected:")
        print(contingency_table.fittedvalues.round(2))
        
    # Combine observed and expected counts
    df_counts = pd.DataFrame(contingency_table.table_orig)
    df_counts = df_counts.reset_index()
    df_counts = df_counts.melt(
        id_vars=categorical_predictor_variable,
        var_name=categorical_outcome_variable,
        value_name='Observed'
    )
    df_temp = pd.DataFrame(contingency_table.fittedvalues.round(2))
    df_temp = df_temp.reset_index()
    df_temp = df_temp.melt(
        id_vars=categorical_predictor_variable,
        var_name=categorical_outcome_variable,
        value_name='Expected'
    )
    df_counts = df_counts.merge(
        df_temp,
        how='left',
        on=[categorical_predictor_variable, categorical_outcome_variable]
    )
    
    # Convert to long-form
    df_counts = df_counts.melt(
        id_vars=[categorical_predictor_variable, categorical_outcome_variable],
        var_name='Value Type',
        value_name='Count'
    )
    
    # Show barplot of observed and expected values, if requested
    if show_plot:
        ax = sns.catplot(
            data=df_counts,
            x=categorical_outcome_variable,
            y='Count',
            hue='Value Type',
            col=categorical_predictor_variable,
            kind='bar',
            height=6,
            palette='Set2'
        )
        ax.fig.subplots_adjust(top=.9)
        ax.fig.text(
            x=0.07,
            y=1,
            s="Observed vs. Expected Counts",
            fontsize=14,
            ha='left',
            va='top'
        )
        ax.fig.text(
            x=0.07,
            y=0.94,
            s=categorical_outcome_variable + " by " + categorical_predictor_variable,
            fontsize=10,
            ha='left',
            va='bottom'
        )
        # Show plot
        plt.show()
        # Clear plot
        plt.clf()
    
    # Pivot value type column before return obs vs exp counts
    df_counts = df_counts.pivot(
        index=[categorical_outcome_variable, categorical_predictor_variable],
        columns='Value Type',
        values='Count')
    df_counts = df_counts.rename_axis(columns = None).reset_index()
    
    # Add observed minus expected column
    df_counts['Difference (observed minus expected)'] = df_counts['Observed'] - df_counts['Expected']
    
    # Return counts 
    return(df_counts)
