# Load packages
library(stringr)
library(ggplot2)
library(dplyr)
library(forcats)

# Declare function
TestLinearRegressionModelCI = function(regression_model,
                                       test_dataset,
                                       confidence_interval=0.95,
                                       print_ci_accuracy=TRUE,
                                       show_ci_performance_plot=TRUE) {
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
  
  # Add flag showing if outcome variable is within CI
  test_dataset$Is.Within.95.CI = ifelse(
    test = (test_dataset$Lower.bound <= test_dataset[[outcome_variable]]) & (test_dataset[[outcome_variable]] <= test_dataset$Upper.bound),
    TRUE,
    FALSE
  )
  
  # Add residual from estimate
  test_dataset$Estimate.residual = test_dataset$Estimate - test_dataset[[outcome_variable]]
  
  # If requested, print performance of CI
  if (print_ci_accuracy) {
    print(paste0(
      "Share of test dataset captured within the model's confidence interval: ",
      round(sum(test_dataset$Is.Within.95.CI) / nrow(test_dataset) * 100, digits = 2), "%"
    ))
  }
  
  # If requested, generate CI performance plot
  ## Add absolute difference
  test_dataset$Estimate.residual.abs = abs(test_dataset$Estimate.residual)
  ## Reorder dataset
  test_dataset = test_dataset %>% arrange(!!sym(outcome_variable))
  ## Add test identifier
  test_dataset$Test.record = as.factor(row.names(test_dataset))
  if (show_ci_performance_plot) {
    p = ggplot(data = test_dataset,
               aes(x = .data$Test.record,
                   y = .data[["Estimate"]])) +
      # Add line connecting observed and estimated values
      geom_linerange(data = subset(test_dataset, Estimate.residual < 0), 
                     aes(x = .data$Test.record, 
                         ymin = .data[["Estimate"]], 
                         ymax = .data[[outcome_variable]]),
                     size = 1,
                     alpha = 0.5,
                     color="#333333") +
      geom_linerange(data = subset(test_dataset, Estimate.residual >= 0), 
                     aes(x = .data$Test.record, 
                         ymin = .data[[outcome_variable]], 
                         ymax = .data[["Estimate"]]),
                     size = 1,
                     alpha = 0.5,
                     color="#333333") +
      # Add points for estimated values
      geom_point(data = test_dataset,
                 aes(x = .data$Test.record,
                     y = .data[["Estimate"]]),
                 color = "#333333",
                 fill = "#333333",
                 alpha = 0.50,
                 size = 1.5) + 
      # Add points for observed values within CI
      geom_point(data = subset(test_dataset, Is.Within.95.CI == TRUE),
                 aes(x = .data$Test.record,
                     y = .data[[outcome_variable]]),
                 color = "#32a852",
                 fill = "#32a852",
                 alpha = 0.50,
                 size = 2) +
      # Add points for observed values outside of CI
      geom_point(data = subset(test_dataset, Is.Within.95.CI == FALSE),
                 aes(x = .data$Test.record,
                     y = .data[[outcome_variable]]),
                 color = "#e84427",
                 fill = "#e84427",
                 alpha = 0.50,
                 size = 2) +
      labs(title = "Model performance",
           subtitle = str_wrap("Green dots represent observed values that were captured in the model's confidence interval, while red dots are values that are outside of the model's confidence interval.", width = 110),
           y = "Estimated and observed values") +
      theme_minimal() +
      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            axis.title.y = element_blank(),
            axis.text.y = element_blank()) +
      coord_flip()
    plot(p)
    test_dataset$Test.record = NULL
    test_dataset$Estimate.residual.abs = NULL
  }
  
  # Return test dataset
  return(test_dataset)
}
