# Load packages
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

# Declare function
def TwoSampleTTestIndependent(dataframe,
                              outcome_column,
                              grouping_column,
                              confidence_interval = .95,
                              alternative_hypothesis = "two-sided",
                              homogeneity_of_variance = True,
                              null_handling = 'omit'):
    # Get groupings (there should only be two)
    list_groups = dataframe[grouping_column].unique()
    if len(list_groups) > 2:
        raise ValueError("T-Test can only be performed with two groups. The grouping column you selected has " + str(len(list_groups)) + " groups.")
    
    # Show pivot table summary
    table_groups = pd.pivot_table(dataframe, 
        values=[outcome_column], 
        index=[grouping_column],
        aggfunc={outcome_column: [len, min, max, np.mean, np.std]}
    )
    print(table_groups)
    print("\n")
    
    # Restructure data
    dataframe = dataframe[[
        outcome_column,
        grouping_column,
    ]]
    group1 = dataframe[dataframe[grouping_column]==list_groups[0]]
    group2 = dataframe[dataframe[grouping_column]==list_groups[1]]
    
    # Conduct two sample t-test
    ttest_results = ttest_ind(
        group1[outcome_column], 
        group2[outcome_column], 
        axis=0,
        nan_policy=null_handling, 
        alternative=alternative_hypothesis,
        equal_var=homogeneity_of_variance
    )

    # Print interpretation of results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of the " + list_groups[0] + " and " + list_groups[1] + " groups.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of the " + list_groups[0] + " group is " + alternative_hypothesis + " than that of the " + list_groups[1] + " group to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(ttest_results.pvalue, 3)))
    print("statistic:", str(round(ttest_results.statistic, 3)))

