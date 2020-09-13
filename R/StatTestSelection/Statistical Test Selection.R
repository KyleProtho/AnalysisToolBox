
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
} else {
     is_continuous_outcome = FALSE
}
if (outcome_variable_type_1 == "Q" & outcome_variable_type_2 == "C") {
     is_discrete_outcome = TRUE
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

# If cateogrical variable selected, determine number of groups ---------------------
prompt_text = paste("You've selected a categorical explanatory variable.",
                    "Next, you'll have to consider the number of groups you are comparing.",
                    "The appropriate statistical test for you analysis most often depends on whether you are comparing 2 or more than 2 groups.",
                    "--  Enter 'Y' if you are comparing 2 groups.",
                    "--  Enter 'N' if you are comparing 3 or more groups.",
                    sep = "\n")
is_two_groups = rstudioapi::showPrompt(
     title = "Are You Comparing 2 Groups?",
     message = prompt_text, 
     default = ""
)
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
                         "--  Transform the outcome (e.g. take the natural log or square root of the outcome) in an effort to make it more normally distributed, then running the T-Test on that transformed variable. Just remember to transform the estimates back to their original form, or else you'll confuse your reader.",
                         "--  Run a Wilcoxon signed-rank test, which compares the ranks of the values in each group rather than the absolute values. This is a non-parametric test though, so you will not get any estimates on the outcome - you'll only get a p-value.",
                         sep = "\n")
     rstudioapi::showQuestion(
          title = "Statistical Test Recommendation",
          message = prompt_text)
}

# ANOVA (and alternatives) -------------------------------------------------

