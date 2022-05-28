# Load packages

# Declare function
CalculateCoefficentOfVariation = function(dataframe,
                                          column_name) {
  # Calculate coefficient of variation
  value_mean = mean(dataframe[[column_name]])
  value_sd = sd(dataframe[[column_name]])
  coef_of_variation = value_sd / value_mean
  
  # Return coefficient of variation
  return(coef_of_variation)
}
