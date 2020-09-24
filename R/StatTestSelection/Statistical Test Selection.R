
# Load packages -----------------------------------------------------------


# Determine outcome variable type -----------------------------------------
prompt_text = paste("First, we need to examine the data type of the outcome you're analyzing.",
                    "--  Enter 'Q' if your outcome is quantitative. Quantitative variables are numbers that can be easily added, subtracted, multiplied, or divided.",
                    "--  Enter 'C' if your outcome is categorical. Categorical variables can be either text or a number that represents one of a limited, fixed set of qualitative values. Examples included alive/dead, high/moderate/low, success/failure, Likert scale values or ratings (1, 2, 3, 4, or 5), etc.",
                    "--  Enter 'T' if your outcome is time-to-event. Time-to-event variables represent the length of time until the occurrence of a well-defined event of interest. Examples include the number of days until readmission or the number of months a customer remains subscribed to their local newspaper before canceling.",
                    sep = "\n")
outcome_variable_type_1 = rstudioapi::showPrompt(
     title = "Enter Outcome Variable Type",
     message = prompt_text, 
     default = ""
)
if (outcome_variable_type_1 == "C") {
        is_categorical_outcome = TRUE
} else {
        is_categorical_outcome = FALSE
}

# If quantitative outcome selected, determine discrete or continuous --------------
if (outcome_variable_type_1 == "Q") {
     prompt_text = paste("You've selected a quantitative outcome.",
                         "Next, you should specify what type of quantitative variable you have.",
                         "--  Enter 'C' if your outcome is continuous. Continuous variables are numbers that can theoretically take any value - it's only limited by how precisely you can measure it. Examples include a person's height, blood sugar levels, or a country's GDP.",
                         "--  Enter 'D' if your outcome is discrete. Discrete variables can only take on certain values, such as counts.",
                         sep = "\n")
     outcome_variable_type_2 = rstudioapi::showPrompt(
          title = "Enter Quantitative Outcome Type",
          message = prompt_text, 
          default = ""
     )
}
if (outcome_variable_type_1 == "Q" & outcome_variable_type_2 == "C") {
     is_continuous_outcome = TRUE
     is_quantitative_outcome = TRUE
} else {
     is_continuous_outcome = FALSE
}
if (outcome_variable_type_1 == "Q" & outcome_variable_type_2 == "D") {
     is_discrete_outcome = TRUE
     is_quantitative_outcome = TRUE
} else {
     is_discrete_outcome = FALSE
}

# If categorical outcome selected, determine nominal or ordinal -------------------
if (outcome_variable_type_1 == "C") {
     prompt_text = paste("You've selected a categorical outcome.",
                         "Next, you should specify what type of categorical variable you have.",
                         "--  Enter 'N' if your outcome is nominal. Nominal variables are unordered cateories, meaning there is no hierarchy, preference, or some other element of comparison among them.",
                         "--  Enter 'O' if your outcome is ordinal. Ordinal variables are ordered categories, meaning there is a hierarchy or relative value among them. In other words, these categories can be sorted in some way other than alphabetically.",
                         sep = "\n")
     outcome_variable_type_2 = rstudioapi::showPrompt(
          title = "Enter Cateogorical Outcome Type",
          message = prompt_text, 
          default = ""
     )
}
if (outcome_variable_type_1 == "C" & outcome_variable_type_2 == "N") {
     is_nominal_outcome = TRUE
} else {
     is_nominal_outcome = FALSE
}
if (outcome_variable_type_1 == "C" & outcome_variable_type_2 == "O") {
     is_ordinal_outcome = TRUE
} else {
     is_ordinal_outcome = FALSE
}

# If time-to-event outcome selected, set type_2 as time-to-event -------------------
if (outcome_variable_type_1 == "T") {
     is_time_to_event_outcome = TRUE
} else {
     is_time_to_event_outcome = FALSE
}

# Determine explanatory variable type ----------------------------------------------
prompt_text = paste("Now that your outcome variable has been considered, you need to consider the independent/explanatory variable of your analysis.",
                    "--  Enter 'C' if your explanatory variable is categorical. Categorical variables can be either text or a number that represents one of a limited, fixed set of qualitative values. Select this one if you are comparing your outcome across groups. Examples included treated/untreated, high/moderate/low, success/failure, Likert scale values or ratings (1, 2, 3, 4, or 5), etc.",
                    "--  Enter 'Q' if your explanatory variable is quantitative. Quantitative variables are numbers that can be easily added, subtracted, multiplied, or divided.",
                    "--  Enter 'M' if you are exploring the explanatory or predictive relationship between many variables and your outcome.",
                    sep = "\n")
explanatory_variable_type_1 = rstudioapi::showPrompt(
     title = "Enter Explanatory Variable Type",
     message = prompt_text, 
     default = ""
)
if (explanatory_variable_type_1 == "C") {
     is_groups = TRUE
} else {
     is_groups = FALSE
}

# If categorical explanatory variable selected, determine number of groups if outcome is quantitative ---------------------
if (explanatory_variable_type_1 == "C" & outcome_variable_type_1 == "Q") {
        prompt_text = paste("You've selected a categorical explanatory variable.",
                            "Next, you'll have to consider the number of groups you are comparing.",
                            "The appropriate statistical test for your analysis most often depends on whether you are comparing 2 or more than 2 groups.",
                            "--  Enter 'Y' if you are comparing 2 groups.",
                            "--  Enter 'N' if you are comparing 3 or more groups.",
                            sep = "\n")
        is_two_groups = rstudioapi::showPrompt(
                title = "Are You Comparing 2 Groups?",
                message = prompt_text, 
                default = ""
        )
}
if (is_two_groups == "Y") {
     is_two_groups = TRUE
} else {
     is_two_groups = FALSE
}

# Determine independent or related observations ------------------------------------
prompt_text = paste("Finally, you will have to specify whether your observations are independent or correlated.",
                    "--  Enter 'I' if your observations are independent. This is most likely the case if your groups or measurements are taken on completely separate subjects or people.",
                    "--  Enter 'R' if your observations are related. For example, if your observations are before/after measurements on the same subjects or people, then your observations are related.",
                    sep = "\n")
observation_type = rstudioapi::showPrompt(
     title = "Enter Type of Observations",
     message = prompt_text, 
     default = ""
)
if (observation_type == "R") {
     is_related_obs = TRUE
} else {
     is_related_obs = FALSE
}



# Show recommendations ----------------------------------------------------
# T-Test (and alternatives) ------------------------------------------------------------------
if (is_continuous_outcome & is_groups & is_two_groups & is_related_obs == FALSE) {
        prompt_text = paste("Since you are comparing a continuous outcome across 2 independent groups, the T-Test might be best for you.",
                            "However, if one of the following scenarios is true, then you may get misleading results in your T-Test:",
                            "--  Sample size is small (less than 100 per group) AND the outcome is not normally distributed; or",
                            "--  Sample sizes in each group are unequal AND variances are non-homogenous. Note that this is only applicable if you want to use pooled variance ('var.equal = TRUE' in the t.test function in R), which you should use whenever possible.",
                            "",
                            "If one of the above scenarios is true, consider taking one of the following corrective actions:",
                            "--  Transform the outcome (e.g. take the natural log or square root of the outcome) in an effort to make it more normally distributed, then running the T-Test on that transformed variable. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                            "--  Run a Wilcoxon rank-sum test, which compares the ranks of the values in each group rather than the absolute values. This is a non-parametric test though, so you will not get any estimates on the outcome - you'll only get a p-value.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text)
}
if (is_continuous_outcome & is_groups & is_two_groups & is_related_obs) {
     prompt_text = paste("Since you are comparing a continuous outcome across 2 related groups, the Paired T-Test might be best for you.",
                         "However, if one of the following scenarios is true, then you may get misleading results in your Paired T-Test:",
                         "--  Sample size is small (less than 100 per group) AND the outcome is not normally distributed; or",
                         "--  Sample sizes in each group are unequal AND variances are non-homogenous. Note that this is only applicable if you want to use pooled variance ('var.equal = TRUE' in the t.test function in R), which you should use whenever possible.",
                         "",
                         "If one of the above scenarios is true, consider taking one of the following corrective actions:",
                         "--  Transform the outcome (e.g. take the natural log or square root of the outcome) in an effort to make it more normally distributed, then running the Paired T-Test on that transformed variable. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                         "--  Run a Wilcoxon signed-rank test, which compares the ranks of the values in each group rather than the absolute values. This is a non-parametric test though, so you will not get any estimates on the outcome - you'll only get a p-value.",
                         sep = "\n")
     rstudioapi::showQuestion(
          title = "Statistical Test Recommendation",
          message = prompt_text)
}

# ANOVA (and alternatives) -------------------------------------------------
if (is_continuous_outcome & is_groups & is_two_groups == FALSE & is_related_obs == FALSE) {
        prompt_text = paste("Since you are comparing a continuous outcome across 3 or more independent groups, the ANOVA might be best for you.",
                            "However, if one of the following scenarios is true, then you may get misleading results in your ANOVA:",
                            "--  Sample size is small (less than 100 per group) AND the outcome is not normally distributed; or",
                            "--  Sample sizes in each group are unequal AND variances are non-homogenous.",
                            "",
                            "If one of the above scenarios is true, consider taking one of the following corrective actions:",
                            "--  Transform the outcome (e.g. take the natural log or square root of the outcome) in an effort to make it more normally distributed, then running the ANOVA on that transformed variable. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                            "--  Run a Kruskal-Wallis test, which compares the ranks of the values in each group rather than the absolute values. This is a non-parametric test though, so you will not get any estimates on the outcome - you'll only get a p-value.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text)
}
if (is_continuous_outcome & is_groups & is_two_groups == FALSE & is_related_obs) {
        prompt_text = paste("Since you are comparing a continuous outcome across 3 or more related groups, the Repeated Measures ANOVA might be best for you.",
                            "However, if one of the following scenarios is true, then you may get misleading results in your Repeated Measures ANOVA:",
                            "--  Sample size is small (less than 100 per group) AND the outcome is not normally distributed; or",
                            "--  Sample sizes in each group are unequal AND variances are non-homogenous.",
                            "",
                            "If one of the above scenarios is true, consider taking one of the following corrective actions:",
                            "--  Transform the outcome (e.g. take the natural log or square root of the outcome) in an effort to make it more normally distributed, then running the Repeated Measures ANOVA on that transformed variable. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                            "--  Run a Kruskal-Wallis test, which compares the ranks of the values in each group rather than the absolute values. This is a non-parametric test though, so you will not get any estimates on the outcome - you'll only get a p-value.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text)
}

# Chi-square test (and alternatives) ----------------------------------------------------------
if (is_categorical_outcome & is_groups & is_related_obs == FALSE) {
        prompt_text = paste("Since you are comparing a categorical outcome across independent groups, then any of the following are options for you:",
                            "",
                            "A)  Chi Square Test.",
                            "However, if one of your group/outcome counts is less than 5, then the Fisher Exact Test is a better alternative.",
                            "",
                            "B)  Difference in Risk",
                            "",
                            "C)  Relative Risk",
                            "",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}

# McNemar's test (and alternatives) -----------------------------------------
if (is_categorical_outcome & is_groups & is_related_obs) {
        prompt_text = paste("Since you are comparing a categorical outcome across related groups, then the McNemar's Test might be best.",
                            "However, if one of your group/outcome counts is less than 5, then the McNemar's Exact Test is a better alternative.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}

# Rate Ratio --------------------------------------------------------------
if (is_discrete_outcome & is_groups & is_related_obs == FALSE) {
        prompt_text = paste("Since you are comparing a discrete outcome (counts) across independent groups, then Rate Ratios and/or Odds Ratios might be best.",
                            "The 'epitools' package is useful for this.",
                            "A helpful note: When an event is rare (less 10% of sample or population experiences it), then the rate ratio and odds ratio become increasingly similar.",
                            "Otherwise, odds ratios can exaggerate the frequency of some event compared to rate ratios. Keep that in mind when interpreting your results.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}

# Kaplan-Meier statistics --------------------------------------------------------------
if (is_time_to_event_outcome & is_groups & is_related_obs == FALSE) {
        prompt_text = paste("Since you are comparing time-to-event across independent groups, then it is best if you use one of the following methods:",
                            "A)  Kaplan-Meier statistics (also known as Survival Analysis)",
                            "B)  Rate Ratios and/or Odds Ratios (the 'epitools' package is useful for this)",
                            "",
                            "A helpful note: When an event is rare (less 10% of sample or population experiences it), then the rate ratio and odds ratio become increasingly similar.",
                            "Otherwise, odds ratios can exaggerate the frequency of some event compared to rate ratios. Keep that in mind when interpreting your results.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}

# Linear regression ----------------------------------------------
if (is_continuous_outcome & (is_groups == FALSE | explanatory_variable_type_1 == "M") & is_related_obs == FALSE) {
        prompt_text = paste("Since you are analyzing the relationship between 1 continuous outcome (dependent variable) and 1 or more predictor (independent) variables, then it is best if you use one of the following methods:",
                            "",
                            "A)  Pearson's correlation coefficent matrix, if you are not interested in the predictive relationship between the dependent and independent variable(s)",
                            "However, if sample size is small (less than 100) AND the outcome is not normally distributed, then you may get misleading results in your Pearson's correlation coeffiencent.",
                            "If the above scenario is true, consider taking one of the following corrective actions:",
                            "--  Use the non-parametric Spearman's correlation coefficient test instead, which finds the relationship between the ranks of each data point rather than their absoluate values.",
                            "--  Transforming your variable(s) (e.g. take the natural log or square root of them) in an effort to make them more normally distributed, then running the Pearson's correlation coefficent test on that transformed variable. Just remember that the estimates of Pearson's correlation coefficient will be on the transformed variable, not the original values.",
                            "",
                            "B)  Linear regression, if you are interested in the predictive relationship between the dependent and independent variable(s)",
                            "However, if one of the following scenarios is true, then you may get misleading results in your Linear regression:",
                            "--  The relationship between the outcome and the predictors is not linear",
                            "--  The outcome is not distributed normally at each value of the predictors. In other words, the random error does not follow a normal distribution",
                            "--  The variance of the outcome at every value of the predictors is not the same. In other words, there is no homogeneity of variances.",
                            "If any of these scenarios are true, then consider transforming your variable(s) (e.g. take the natural log or square root of them) in an effort to make them more normally distributed, then running the Linear Regression on those transformed variables. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}

# Logistic regression -----------------------------------------------------
if (is_categorical_outcome & (is_groups == FALSE | explanatory_variable_type_1 == "M") & is_related_obs == FALSE) {
        prompt_text = paste("Since you are analyzing the relationship between 1 cateogrical (dependent) outcome and 1 or more quantitative and/or ordered categorical predictor (independent) variables, then Logistic Regression might be best.",
                            "Before proceeding, you must ensure:", 
                            "-- That each of your cateogorical outcomes is dummy-coded -- meaning each cateogory is converted to a separate column and the values in that column include only 0s or 1s",
                            "-- The 'linear in the logit' an assumption is met -- meaning the logit function of the outcome has a linear relationship with the continuous predictor",
                            "",
                            "A helpful note: When an event is rare (less 10% of sample or population experiences it), then the rate ratio and odds ratio become increasingly similar.",
                            "Otherwise, odds ratios can exaggerate the frequency of some event compared to rate ratios. Keep that in mind when interpreting your results.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}
if (is_categorical_outcome & (is_groups == FALSE | explanatory_variable_type_1 == "M") & is_related_obs) {
        prompt_text = paste("Since you are analyzing the relationship between 1 cateogrical (dependent) outcome and 1 or more quantitative and/or ordered categorical predictor (independent) variables, then Conditional Logistic Regression might be best.",
                            "Before proceeding, you must ensure that each of your cateogorical outcomes is dummy-coded -- meaning each cateogory is converted to a separate column and the values in that column include only 0s or 1s",
                            "",
                            "A helpful note: When an event is rare (less 10% of sample or population experiences it), then the rate ratio and odds ratio become increasingly similar.",
                            "Otherwise, odds ratios can exaggerate the frequency of some event compared to rate ratios. Keep that in mind when interpreting your results.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}

# Poisson regression ------------------------------------------------------
if (is_discrete_outcome & (is_groups == FALSE | explanatory_variable_type_1 == "M") & is_related_obs == FALSE) {
        prompt_text = paste("Since you are analyzing the relationship between 1 discrete outcome (dependent variable) and 1 or more predictor (independent) variables, then it is best if you use one of the following methods:",
                            "",
                            "A)  Poisson regression",
                            "However, if one of the following scenarios is true, then you may get misleading results in your Poisson regression:",
                            "--  The relationship between the natural log of the outcome (or incidence rate of the outcome) and the predictors is not linear",
                            "--  The variance of the outcome is significantly greater than the mean of the outcome. In other words, the outcome is 'overdispersed' and does not follow a Poisson distribution.",
                            "If any of these scenarios are true, then consider transforming your variable(s) (e.g. take the natural log or square root of them) in an effort to make them comply with the assumption listed above. Then run a Poisson regression using the transformed variables. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                            "If overdispersion remains, consider running a negative binonial regression instead.",
                            "",
                            "B)  Negative binomial regression",
                            "It is an extension of the Poisson model that fits an extra parameter, called the dispersion parameter, that accounts for overdispersion.",
                            "All beta coefficients are interpreted in the same way.",
                            "However, if the relationship between the natural log of the outcome (or incidence rate of the outcome) and the predictors is not linear, then you may get misleading results in your negative binonial regression.",
                            sep = "\n")
        rstudioapi::showQuestion(
                title = "Statistical Test Recommendation",
                message = prompt_text) 
}


# Clear all selections ----------------------------------------------------
rm(list = ls())
