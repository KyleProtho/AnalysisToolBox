
# Load packages -----------------------------------------------------------
library(datasets)
library(dplyr)
library(ggplot2)
library(devtools)
library(table1)
library(stringr)
library(ggthemes)

# Set function parameters -------------------------------------------------
helper_t_test <- function(data,
                          outcome_column,
                          grouping_column,
                          alternative_hypothesis = "two.sided",
                          test_normality_assumption = FALSE,
                          is_homogeneity_of_variances = FALSE,
                          is_related_observations = FALSE,
                          plot_distribution_of_means = TRUE,
                          plot_difference_in_means = TRUE) {
  # Ensure grouping variable is only 2 --------------------------------------
  is_two_groups = ifelse(
    n_distinct(data[[grouping_column]], na.rm = TRUE) == 2,
    TRUE,
    FALSE
  )
  if (is_two_groups == FALSE) {
    error_string = paste("The T-Test is suitable for 2 groups only.",
                         "The grouping variable you selected has less than, or more than 2 unique values/groups.",
                         "If you are comparing 3 or more groups, try using ANOVA instead.",
                         sep = "\n")
    cat(error_string)
    return(NULL)
  }
  
  # Ensure alternative hypothesis is a valid argument -----------------------
  valid_alternative_hypotheses = c(
    "two.sided",
    "less",
    "greater"
  )
  is_valid_h1 = match(alternative_hypothesis, valid_alternative_hypotheses)
  is_valid_h1 = ifelse(is.na(is_valid_h1),
                       FALSE,
                       TRUE)
  if (is_valid_h1 == FALSE) {
    error_string = paste("The alternative hypothesis you selected is not valid.",
                         "The valid selections are:",
                         "  -- 'two.sided' >> Select this if you want to see if the outcome of the 2 groups differ in any direction.",
                         "  -- 'less' >> Select this if you want to see if the outcome in the reference/control group is less than that of the exposed/treatment group.",
                         "  -- 'greater' >> Select this if you want to see if the outcome in the reference/control group is greater than that of the exposed/treatment group.",
                         sep = "\n")
    cat(error_string)
    return(NULL)
  }
  

  # Test for normality of outcome -------------------------------------------
  if (test_normality_assumption) {
    # Use Scott method to get number of bins
    max_value = max(data[[outcome_column]], na.rm = TRUE)
    min_value = min(data[[outcome_column]], na.rm = TRUE)
    sd_value = sd(data[[outcome_column]], na.rm = TRUE)
    obs_count = nrow(data)
    scott_number_bins_numer = (max_value - min_value) * obs_count ^ (1/3)
    scott_number_bins_denom = 3.5 * sd_value
    scott_number_bins = scott_number_bins_numer / scott_number_bins_denom
    scott_number_bins = round(scott_number_bins, 0)
    # Use Scott method to get bin width
    scott_bin_width_numer = 3.5 * sd_value
    scott_bin_width_denom = obs_count ^ (1/3)
    scott_bin_width = scott_bin_width_numer / scott_bin_width_denom
    scott_bin_width = ifelse(scott_bin_width < 1,
                             1,
                             round(scott_bin_width, 0))
    # Generate title for plot
    title_for_plot = paste("Histogram - Distribution of the Mean for Each Group")
    title_for_plot = str_wrap(title_for_plot, width = 60)
    # Draw plot
    p = ggplot(data, aes(x = data[[outcome_column]])) + 
      geom_histogram(aes(y = ..density..),
                     color = "black",
                     fill = "#7f1a61",
                     bins = scott_number_bins,
                     binwidth = scott_bin_width) +
      geom_density(alpha = .4,
                   fill = "#8c0303") + 
      labs(title = title_for_plot,
           y = "Density",
           x = paste(outcome_column)) + 
      theme_hc() +
      facet_grid(data[[grouping_column]] ~ .)
    plot(p)
    
  }
  
  # Get groups --------------------------------------------------------------
  unique_groups = unique(data[[grouping_column]])
  unique_groups = sort(unique_groups)
  reference_group = unique_groups[1]
  exposed_group = unique_groups[2]
  cat(paste0("Your reference / control group is '", grouping_column, "' = ", reference_group, "\n"))
  cat(paste0("Your exposed / treatment group is '", grouping_column, "' = ", exposed_group, "\n"))
  
  
  # Generate formula string -------------------------------------------------
  formula_string = paste0(outcome_column, 
                          " ~ ", 
                          grouping_column)
  formula_string = as.formula(formula_string)
  
  # Conduct T-Test ----------------------------------------------------------
  test_results = t.test(formula_string,
                        data = data,
                        alternative = alternative_hypothesis,
                        var.equal = is_homogeneity_of_variances,
                        paired = is_related_observations)
  
  # For each group, generate distribution -----------------------------------
  trials = 10000
  df_results = data.frame()
  for (group in unique_groups) {
    # Filter to current group
    df_temp = filter(
      data,
      !!sym(grouping_column) == group
    )
    # Set sample size for random sampling
    sample_size = nrow(df_temp) * .20
    if (sample_size < 10) {
      sample_size = 10
    } else if (sample_size < 25) {
      sample_size = 25
    } else if (sample_size < 50) {
      sample_size = 50
    }
    else {
      sample_size = 100
    }
    # Simulate mean of random sample
    sim_distribution = c()
    for (variable in 1:trials) {
      trial_results = sample(c(df_temp[[outcome_column]]),
                             size = 10,
                             replace = TRUE)
      mean_trial = mean(trial_results)
      sim_distribution = append(x = sim_distribution,
                                values = mean_trial)
    }
    # Save means to dataframe
    df_sim = data.frame(trials = 1:10000)
    df_sim$means = sim_distribution
    df_sim$group = group
    df_results = bind_rows(
      df_results,
      df_sim
    )
    rm(df_sim)
  }
  
  # Generate plot for distribution of the means -----------------------------
  if (plot_distribution_of_means) {
    # Use Scott method to get number of bins
    max_value = max(df_results[["means"]], na.rm = TRUE)
    min_value = min(df_results[["means"]], na.rm = TRUE)
    sd_value = sd(df_results[["means"]], na.rm = TRUE)
    obs_count = nrow(df_results)
    scott_number_bins_numer = (max_value - min_value) * obs_count ^ (1/3)
    scott_number_bins_denom = 3.5 * sd_value
    scott_number_bins = scott_number_bins_numer / scott_number_bins_denom
    scott_number_bins = round(scott_number_bins, 0)
    # Use Scott method to get bin width
    scott_bin_width_numer = 3.5 * sd_value
    scott_bin_width_denom = obs_count ^ (1/3)
    scott_bin_width = scott_bin_width_numer / scott_bin_width_denom
    scott_bin_width = ifelse(scott_bin_width < 1,
                             1,
                             round(scott_bin_width, 0))
    # Generate title for plot
    title_for_plot = paste("Histogram - Distribution of the Mean for Each Group")
    title_for_plot = str_wrap(title_for_plot, width = 60)
    # Draw plot
    p = ggplot(df_results, aes(x = df_results[["means"]])) + 
      geom_histogram(aes(y = ..density..),
                     color = "black",
                     fill = "#7f1a61",
                     bins = scott_number_bins,
                     binwidth = scott_bin_width) +
      geom_density(alpha = .4,
                   fill = "#8c0303") + 
      labs(title = title_for_plot,
           y = "Density",
           x = paste("Mean of", outcome_column)) + 
      theme_hc() +
      facet_grid(df_results[["group"]] ~ .)
    plot(p)
    
  }
  
  # Create restructured dataframe with ref and exp means joined -------------
  df_ref = filter(
    df_results,
    group == 0
  )
  df_ref$group = NULL
  df_ref$ref_mean = df_ref$means
  df_ref$means = NULL
  df_exp = filter(
    df_results,
    group == 1
  )
  df_exp$group = NULL
  df_exp$exp_mean = df_exp$means
  df_exp$means = NULL
  df_results_2 = left_join(
    df_ref,
    df_exp,
    by = "trials"
  )
  rm(df_ref,
     df_exp)
  
  
  # Add null hypothesis means -----------------------------------------------
  sim_distribution = c()
  # Filter to reference group
  df_temp = filter(
    data,
    !!sym(grouping_column) == reference_group
  )
  trials = 10000
  for (variable in 1:trials) {
    trial_results = sample(c(df_temp[[outcome_column]]),
                           size = 10,
                           replace = TRUE)
    mean_trial = mean(trial_results)
    sim_distribution = append(x = sim_distribution,
                              values = mean_trial)
  }
  # Save means to dataframe
  df_sim = data.frame(trials = 1:10000)
  df_sim$ref_means_2 = sim_distribution
  df_results_2 = left_join(
    df_results_2,
    df_sim,
    by = "trials"
  )
  rm(df_sim)
  rm(df_temp)
  
  # Get null distribution (no difference) and observed -----------------------------------
  df_results_2$null_difference = df_results_2$ref_mean - df_results_2$ref_means_2
  df_results_2$difference = df_results_2$ref_mean - df_results_2$exp_mean
  
  # Generate p-value plot ---------------------------------------------------
  if (plot_difference_in_means) {
    # Use Scott method to get number of bins
    max_value = max(df_results_2[["null_difference"]], na.rm = TRUE)
    min_value = min(df_results_2[["null_difference"]], na.rm = TRUE)
    sd_value = sd(df_results_2[["null_difference"]], na.rm = TRUE)
    obs_count = nrow(data)
    scott_number_bins_numer = (max_value - min_value) * obs_count ^ (1/3)
    scott_number_bins_denom = 3.5 * sd_value
    scott_number_bins = scott_number_bins_numer / scott_number_bins_denom
    scott_number_bins = round(scott_number_bins, 0)
    # Use Scott method to get bin width
    scott_bin_width_numer = 3.5 * sd_value
    scott_bin_width_denom = obs_count ^ (1/3)
    scott_bin_width = scott_bin_width_numer / scott_bin_width_denom
    scott_bin_width = ifelse(scott_bin_width < 1,
                             1,
                             round(scott_bin_width, 0))
    # Generate title for plot
    title_for_plot = paste("Histogram - Difference in Means Between Groups")
    title_for_plot = str_wrap(title_for_plot, width = 60)
    # Calculate difference in means for vertical line
    observed_difference_in_means = test_results[["estimate"]][1] - test_results[["estimate"]][2]
    # Draw plot
    p = ggplot(df_results_2, 
               aes(x = df_results_2[["null_difference"]]))
    # If two-sided test, insert two lines. Otherwise, insert one line
    if (alternative_hypothesis == "two.sided") {
      p = p + 
        geom_vline(xintercept = observed_difference_in_means,
                   color = "#de0000",
                   size = 2) +
        geom_vline(xintercept = observed_difference_in_means * -1,
                   color = "#de0000",
                   size = 2)
      abs_observed_difference_in_means = abs(observed_difference_in_means)
      p = p + 
        annotate("rect",
                 xmin = -Inf, 
                 xmax = abs_observed_difference_in_means * -1,
                 ymin = 0, 
                 ymax = Inf, 
                 alpha = 0.2,
                 fill = "#de0000") + 
        annotate("rect",
                 xmin = abs_observed_difference_in_means, 
                 xmax = Inf,
                 ymin = 0, 
                 ymax = Inf, 
                 alpha = 0.2,
                 fill = "#de0000")
    } else {
      p = p +
        geom_vline(xintercept = observed_difference_in_means,
                   color = "#de0000",
                   size = 2)
      if (alternative_hypothesis == "greater") {
        p = p + annotate("rect",
                         xmin = observed_difference_in_means, 
                         xmax = Inf, 
                         ymin = 0, 
                         ymax = Inf, 
                         alpha = 0.2,
                         fill = "#de0000")
      } else {
        p = p + annotate("rect",
                         xmin = -Inf, 
                         xmax = observed_difference_in_means,
                         ymin = 0, 
                         ymax = Inf, 
                         alpha = 0.2,
                         fill = "#de0000")
      }
      
    }
    p = p + 
      geom_histogram(color = "black",
                     fill = "#7f1a61",
                     bins = scott_number_bins,
                     binwidth = scott_bin_width) +
      labs(title = title_for_plot,
           subtitle = "Reference/Control group minus Exposed/Treatement Group",
           caption = paste("This chart shows what one could expect if there were no detectable difference between the means of the groups (the null hypothesis).",
                           "The p-value, which represents P(your data | the null hypothesis being true), is illustrated by the red line(s).",
                           sep = "\n"),
           y = "Simulated Occurences of the Mean",
           x = "Difference in Means between Groups") + 
      theme_hc()
    plot(p)
  }
  
  # Return T-test results ---------------------------------------------------
  return(test_results)
}


# TESTING / DEMO ------------------------------------------------------
df_mtcars = mtcars
t_test_results_am = helper_t_test(data = df_mtcars,
                                  outcome_column = "mpg",
                                  grouping_column = "am",
                                  test_normality_assumption = TRUE,
                                  plot_distribution_of_means = TRUE,
                                  plot_difference_in_means = TRUE)
t_test_results_am_less = helper_t_test(data = df_mtcars,
                                       outcome_column = "mpg",
                                       grouping_column = "am",
                                       alternative_hypothesis = "less",
                                       plot_distribution_of_means = FALSE,
                                       plot_difference_in_means = TRUE)
t_test_results_am_greater = helper_t_test(data = df_mtcars,
                                          outcome_column = "mpg",
                                          grouping_column = "am",
                                          alternative_hypothesis = "greater",
                                          plot_distribution_of_means = FALSE,
                                          plot_difference_in_means = TRUE)
t_test_results_cyl = helper_t_test(data = df_mtcars,  ## This should fail since there are more than two groups
                                   outcome_column = "mpg",
                                   grouping_column = "cyl",
                                   plot_distribution_of_means = TRUE,
                                   plot_difference_in_means = TRUE)
