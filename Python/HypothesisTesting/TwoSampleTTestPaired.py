# Load packages
from scipy.stats import ttest_rel
import pandas as pd
import numpy as np

# Declare function
def TwoSampleTTestPaired(dataframe,
                         outcome1_column,
                         outcome2_column,
                         confidence_interval = .95,
                         alternative_hypothesis = "two-sided",
                         null_handling = 'omit'):
    # Show pivot table summary
    table_groups = dataframe.groupby(
        [True]*len(dataframe)
    ).agg(
        {
            outcome1_column: [min, max, np.mean, np.std],
            outcome2_column: [min, max, np.mean, np.std]
        }
    )
    print(table_groups)
    print("\n")
    
    # Conduct two sample t-test
    ttest_results = ttest_rel(
        dataframe[outcome1_column], 
        dataframe[outcome2_column], 
        axis=0,
        nan_policy=null_handling, 
        alternative=alternative_hypothesis
    )

    # Print interpretation of results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of " + outcome1_column + " and " + outcome2_column + ".")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of " + outcome1_column + " is " + alternative_hypothesis + " than that of " + outcome2_column + " to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(ttest_results.pvalue, 3)))
    print("statistic:", str(round(ttest_results.statistic, 3)))
