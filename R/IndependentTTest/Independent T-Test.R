
# Load packages -----------------------------------------------------------
library(datasets)
library(dplyr)
library(ggplot2)
library(devtools)

# Build test dataset ------------------------------------------------------
df_mtcars = mtcars

# Set function parameters -------------------------------------------------
data = df_mtcars
outcome_column = "mpg"
grouping_column = "am"
alternative_hypothesis = "two.sided"
is_homogeneity_of_variances = FALSE
is_related_observations = FALSE

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


# Import histogram function from Github -----------------------------------
source_url("https://raw.github.com/tonybreyal/Blog-Reference-Functions/master/R/bingSearchXScraper/bingSearchXScraper.R")

# Create null distribution ------------------------------------------------
trials = 10000
null_distribution = c()
for (variable in 1:trials) {
  trial_results = rnorm(n = 10,
                        mean = 0,
                        sd = test_results$stderr)
  mean_trial = mean(trial_results)
  null_distribution = append(x = null_distribution,
                             values = mean_trial)
}

# Create alternative distribution -----------------------------------------

