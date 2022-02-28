
# Load packages -----------------------------------------------------------
library(datasets)
library(dplyr)
library(ggplot2)
library(devtools)
library(table1)
library(stringr)
library(ggthemes)

# Set function parameters -------------------------------------------------
ConductTTest = function(dataframe,
                        outcome_column,
                        grouping_column,
                        alternative_hypothesis = "two.sided",
                        test_normality_assumption = FALSE,
                        test_homogeneity_of_variances_assumption = FALSE,
                        is_homogeneity_of_variances = FALSE,
                        is_related_observations = FALSE,
                        plot_distribution_of_means = TRUE) {
  # Ensure grouping variable is only 2 --------------------------------------
  is_two_groups = ifelse(
    n_distinct(dataframe[[grouping_column]], na.rm = TRUE) == 2,
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
  # Make caption string
  caption_string = paste(
    "The normality assumption requires that the outcome is normally distributed (mostly) within groups.",
    "However, if your sample size is not too small (>= 100), then you do not need worry too much about meeting this assumption.",
    sep = "\n")
  # Generate plot
  if (test_normality_assumption) {
    # Generate title for plot
    title_for_plot = paste("Examining the Normality Assumption")
    title_for_plot = str_wrap(title_for_plot, width = 80)
    # Draw plot
    p = ggplot(data = dataframe, 
               aes(x = .data[[outcome_column]])) + 
      geom_density(alpha = .4,
                   fill = "#8c0303") + 
      labs(title = title_for_plot,
           subtitle = str_wrap(caption_string, width = 110),
           y = "Density",
           x = paste(outcome_column)) + 
      theme_minimal() +
      theme(panel.grid.major=element_blank(),
            panel.grid.minor=element_blank()) +
      facet_grid(.data[[grouping_column]] ~ .)
    plot(p)
  }
  
  # Test for homogeneity of variances ----------------------------------------
  if (test_homogeneity_of_variances_assumption) {
    # Make caption string
    caption_string = paste(
      "The homogeneity of variances assumption requires that the variance of the outcome is the same in the different groups.",
      "However, if sample sizes in the groups are similar, then you do not need worry too much about meeting this assumption.",
      sep = "\n")
    # Generate plot
    p = ggplot(data = dataframe,
               aes(x = as.factor(.data[[grouping_column]]),
                   y = .data[[outcome_column]],
                   fill = as.factor(.data[[grouping_column]]))) + 
      geom_boxplot() + 
      geom_dotplot(binaxis = 'y', 
                   stackdir = 'center', 
                   dotsize = .5) +
      labs(title = "Examining the Homogeneity of Variances Assumption",
           subtitle = str_wrap(caption_string, width = 110),
           x = "Groups",
           y = outcome_column) + 
      theme_minimal() + 
      theme(legend.position = "none") 
    plot(p)
  }
  
  # Get groups --------------------------------------------------------------
  unique_groups = unique(dataframe[[grouping_column]])
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
                        data = dataframe,
                        alternative = alternative_hypothesis,
                        var.equal = is_homogeneity_of_variances,
                        paired = is_related_observations)
  
  # For each group, generate distribution -----------------------------------
  trials = 10000
  df_results = data.frame()
  for (group in unique_groups) {
    # Filter to current group
    df_temp = filter(
      dataframe,
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
    # Generate title for plot
    title_for_plot = paste("Distribution of the Mean for Each Group")
    title_for_plot = str_wrap(title_for_plot, width = 60)
    # Draw plot
    p = ggplot(data = df_results, aes(x = .data[["means"]])) + 
      geom_histogram(aes(y = ..density..),
                     color = "black",
                     fill = "#7f1a61") +
      geom_density(alpha = .4,
                   fill = "#8c0303") + 
      labs(title = title_for_plot,
           y = "Density",
           x = paste("Mean of", outcome_column)) + 
      theme_minimal() +
      theme(panel.grid.major=element_blank(),
            panel.grid.minor=element_blank()) +
      facet_grid(df_results[["group"]] ~ .)
    plot(p)
    
  }
  
  # Return T-test results ---------------------------------------------------
  return(test_results)
}
