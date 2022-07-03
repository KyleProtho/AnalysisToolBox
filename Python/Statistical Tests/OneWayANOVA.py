# Load packages
from statsmodels.stats.oneway import anova_oneway
import pandas as pd
import numpy as np

# Declare function
def OneWayANOVA(dataframe,
                outcome_column,
                grouping_column,
                signficance_level = 0.05,
                homogeneity_of_variance = True,
                use_welch_correction = False):
    # Show pivot table summary
    table_groups = pd.pivot_table(dataframe, 
        values=[outcome_column], 
        index=[grouping_column],
        aggfunc={outcome_column: [len, min, max, np.mean, np.std, np.var]}
    )
    print(table_groups)
    print("\n")

    # Set variance argument
    if homogeneity_of_variance:
        variance_arg = "equal"
    else:
        variance_arg = "unequal"

    # Conduct one-way ANOVA
    test_results = anova_oneway(
        data=dataframe[outcome_column], 
        groups=dataframe[grouping_column], 
        use_var=variance_arg, 
        welch_correction=use_welch_correction
    )

    # Print interpretation of results
    if test_results.pvalue <= signficance_level:
        print("There is a statistically detectable difference between the sample means of at least 2 groups.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(test_results.pvalue, 3)))
    print("degrees of freedom between groups:", str(int(test_results.df_num)))
    print("degrees of freedom across all groups:", str(int(test_results.df_denom)))
    print("f-statistic:", str(round(test_results.statistic, 3)))
