# Load packages
from scipy.stats import ttest_ind_from_stats

# Declare function
def TTestOfTwoMeansFromStats(group1_mean,
                             group1_sd,
                             group1_sample_size,
                             group2_mean,
                             group2_sd,
                             group2_sample_size,
                             homogeneity_of_variance = True,
                             confidence_interval = 0.95,
                             alternative_hypothesis = "two-sided"):
        
    # Conduct t-test
    ttest_results = ttest_ind_from_stats(
        mean1 = group1_mean, 
        std1 = group1_sd, 
        nobs1 = group1_sample_size,
        mean2 = group2_mean, 
        std2 = group2_sd, 
        nobs2 = group2_sample_size,
        equal_var = homogeneity_of_variance,
        alternative = alternative_hypothesis
    )

    # Interpret test results
    if alternative_hypothesis == "two-sided" and ttest_results.pvalue <= (1-confidence_interval):
        print("There is a statistically detectable difference between the sample means of Group 1 and Group 2.")
    elif ttest_results.pvalue <= (1-confidence_interval):
        print("The sample mean of Group 1 is " + alternative_hypothesis + " than that of Group 2 to a statistically detectable extent.")
    else:
        print("Cannot reject the null hypothesis.")
    print("p-value:", str(round(ttest_results.pvalue, 3)))
    print("statistic:", str(round(ttest_results.statistic, 3)))
