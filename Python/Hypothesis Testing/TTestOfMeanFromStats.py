# Load packages 
from scipy import stats
from math import sqrt

# Declare function
def TTestOfMeanFromStats(sample_mean,
                         sample_sd,
                         sample_size,
                         hypothesized_mean,
                         alternative_hypothesis = "two-sided",
                         confidence_interval = .95):
    # If alternative hypothesis is "two-sided", then divide and add half of complement
    if alternative_hypothesis == "two-sided":
        ppf_threshold = confidence_interval + ((1 - confidence_interval) / 2)
    else:
        ppf_threshold = confidence_interval

    # If sample size 30 or more, calculate z-stat. Otherwise, calculate t-stat
    if sample_size < 30:
        test_threshold = stats.t.ppf(ppf_threshold)
    else:
        test_threshold = stats.norm.ppf(ppf_threshold)

    # Calculate difference between sample and hypothesized mean
    diff_in_means = hypothesized_mean - sample_mean

    # Calculate standard error of sample
    standard_error = sample_sd / sqrt(sample_size)

    # Calculate test statistic
    test_stat = diff_in_means / standard_error

    # Interpret results
    if alternative_hypothesis == "two-sided" and abs(test_stat) > test_threshold:
        print("There is a statistically detectable difference between the hypothesized mean and the sample mean.")
        if sample_size < 30:
            print("p-value:", str(round(stats.t.cdf(abs(test_stat)*-1), 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            print("p-value:", str(round(stats.norm.cdf(abs(test_stat)*-1), 3)))
            print("statistic:", str(round(test_stat, 3)))
    elif alternative_hypothesis == "less" and test_stat < test_threshold*-1:
        print("The hypothesized mean is", alternative_hypothesis, "than the sample mean to a statistically detectable extent.")
        if sample_size < 30:
            print("p-value:", str(round(stats.t.cdf(abs(test_stat)*-1), 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            print("p-value:", str(round(stats.norm.cdf(abs(test_stat)*-1), 3)))
            print("statistic:", str(round(test_stat, 3)))
    elif alternative_hypothesis == "greater" and test_stat > test_threshold:
        print("The hypothesized mean is", alternative_hypothesis, "than the sample mean to a statistically detectable extent.")
        if sample_size < 30:
            print("p-value:", str(round(1-stats.t.cdf(test_stat), 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            print("p-value:", str(round(1-stats.norm.cdf(test_stat), 3)))
            print("statistic:", str(round(test_stat, 3)))
    else:
        print("Cannot reject the null hypothesis")
        if sample_size < 30:
            if alternative_hypothesis != "greater":
                print("p-value:", str(round(stats.t.cdf(test_stat), 3)))
            else:
                print("p-value:", str(round(1-stats.t.cdf(test_stat), 3)))
            print("statistic:", str(round(test_stat, 3)))
        else:
            if alternative_hypothesis != "greater":
                print("p-value:", str(round(stats.norm.cdf(test_stat), 3)))
            else:
                print("p-value:", str(round(1-stats.norm.cdf(test_stat), 3)))
            print("statistic:", str(round(test_stat, 3)))
