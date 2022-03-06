# Load packages
library(stringr)
library(ggplot2)
library(dplyr)
library(forcats)

# Declare function
TestLogisticRegressionModelCI = function(regression_model,
                                         test_dataset,
                                         confidence_interval=0.95) {
  # Get outcome variable name
  outcome_variable = regression_model$terms[[2]]
  
  # Get list of predictors
  list_of_predictors = regression_model[["model"]]
  list_of_predictors = colnames(list_of_predictors)
  list_of_predictors = list_of_predictors[list_of_predictors != outcome_variable]
  
  # Select columns to use in model test
  list_of_variables = append(x = list_of_predictors,
                             values = outcome_variable)
  test_dataset = test_dataset %>% 
    select_(
      .dots = list_of_variables
    )
  
  # Keep complete cases only
  test_dataset = na.omit(test_dataset)
  
  # Get lower, upper, and best Intercept
  test_dataset$Lower.bound = confint(regression_model, '(Intercept)', level=confidence_interval)[1]
  test_dataset$Estimate = regression_model[["coefficients"]][["(Intercept)"]]
  test_dataset$Upper.bound = confint(regression_model, '(Intercept)', level=confidence_interval)[2]
  
  # For each predictor, get the lower, upper, and best beta coef.
  for (variable in list_of_predictors) {
    # Lower bound
    lower_beta = confint(regression_model, variable, level=confidence_interval)[1]
    test_dataset$Lower.bound = test_dataset$Lower.bound + (test_dataset[[variable]] * lower_beta)
    # Estimate
    estimate_beta = regression_model[["coefficients"]][[variable]]
    test_dataset$Estimate = test_dataset$Estimate + (test_dataset[[variable]] * estimate_beta)
    # Upper bound
    upper_beta = confint(regression_model, variable, level=confidence_interval)[2]
    test_dataset$Upper.bound = test_dataset$Upper.bound + (test_dataset[[variable]] * upper_beta)
  }
  
  # Convert logits to probability
  ## Convert to odds ratio
  test_dataset$Lower.bound = exp(test_dataset$Lower.bound)
  test_dataset$Estimate = exp(test_dataset$Estimate)
  test_dataset$Upper.bound = exp(test_dataset$Upper.bound)
  ## Convert to probability
  test_dataset$Lower.bound = test_dataset$Lower.bound /(1 + test_dataset$Lower.bound)
  test_dataset$Estimate = test_dataset$Estimate /(1 + test_dataset$Estimate)
  test_dataset$Upper.bound = test_dataset$Upper.bound /(1 + test_dataset$Upper.bound)
  
  # Return test dataset
  return(test_dataset)
}
