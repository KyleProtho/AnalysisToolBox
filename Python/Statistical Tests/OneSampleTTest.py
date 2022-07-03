# Load packages
from scipy.stats import ttest_1samp
import pandas as pd

# Declare function
def OneSampleTTest(dataframe, 
                   column_name, 
                   hypothesized_value,
                   confidence_interval = .95, 
                   alternative_hypothesis = "two-sided",
                   null_handling = 'omit'):
    # Conduct one sample t-test
    ttest_results = ttest_1samp(
        dataframe[column_name], 
        popmean=hypothesized_value, 
        axis=0, 
        nan_policy=null_handling, 
        alternative=alternative_hypothesis
    )

    # Print interpretation of results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the hypothesized mean and the sample mean.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean is " + alternative_hypothesis + " than the hypothesized mean to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(ttest_results.pvalue, 3)))
    print("statistic:", str(round(ttest_results.statistic, 3)))
