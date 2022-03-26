# Load packages
library(dplyr)
library(Hmisc)

# Declare function
IdentifyCorrelatedPredictors = function(dataframe,
                                        outcome_column,
                                        predictor_columns_to_consider,
                                        top_n_predictors = NULL) {
  # Select outcome and predictor columns 
  dataframe = dataframe %>% 
    select(
      !!sym(outcome_column),
      one_of(predictor_columns_to_consider)
    )
  
  # Keep complete cases in dataframe
  dataframe = dataframe[complete.cases(dataframe), ]
  
  # If top_n_predictors, not specified, use all predictors
  if (is.null(top_n_predictors)) {
    top_n_predictors = length(predictor_columns_to_consider)
  }
  
  # Get correlation coefficients
  corr_coef = rcorr(as.matrix(dataframe))
  
  # Keep "upper triangle" of corr_coef
  corr_coef_upper = upper.tri(corr_coef$r)
  
  # Convert correlation matrix to dataframe with p-values
  df_correlations = data.frame(
    independent_variable = rownames(corr_coef$r)[col(corr_coef$r)[corr_coef_upper]],
    outcome_variable = rownames(corr_coef$r)[row(corr_coef$r)[corr_coef_upper]],
    correlation_coefficient = (corr_coef$r)[corr_coef_upper],
    p_value = corr_coef[["P"]][corr_coef_upper],
    sample_size = corr_coef[["n"]][corr_coef_upper]
  )
  
  # Filter to outcome_variable, arrange by abs value of corr_coef and p-value, and limit to top_n_predictors
  df_correlations = df_correlations %>%
    filter(
      outcome_variable == outcome_column
    ) %>%
    arrange(
      -abs(correlation_coefficient),
      p_value
    )
  df_correlations = head(df_correlations, top_n_predictors)
  
  # Return results
  return(df_correlations)
}

# Test
data(airquality)
results = IdentifyCorrelatedPredictors(dataframe = airquality,
                                       outcome_column = "Temp",
                                       predictor_columns_to_consider = c("Ozone", "Solar.R", "Wind"))

