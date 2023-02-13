# Load packages
library(rmetalog)

# Declare function
CreateMetalogDistribution = function(dataframe,
                                  column_name,
                                  boundary_setting = "u",
                                  upper_boundary = NULL,
                                  lower_boundary = NULL,
                                  plot_fitted_distributions = TRUE) {
  # Set boundary if none specified
  if (boundary_setting == "b" & is.null(upper_boundary)) {
    upper_boundary = max(dataframe[[column_name]])
  }
  if (boundary_setting == "b" & is.null(lower_boundary)) {
    lower_boundary = min(dataframe[[column_name]])
  }
  
  # Convert to numeric (in case function called from Python)
  if (!is.null(upper_boundary)) {
    upper_boundary = as.numeric(upper_boundary)
  }
  if (!is.null(lower_boundary)) {
    lower_boundary = as.numeric(lower_boundary)
  }
  
  # Fit metalog distribution
  if (boundary_setting == "u") {
    metalog_dist = metalog(
      dataframe[[column_name]],
      boundedness = boundary_setting
    )
  } else {
    metalog_dist = metalog(
      dataframe[[column_name]],
      bounds = c(lower_boundary, upper_boundary),
      boundedness = boundary_setting
    )
  }
  
  # Show summary of fitted distribution
  summary(metalog_dist)
  
  # Return metalog distribution object
  return(metalog_dist)
}

# # Test function
# # metalog_petal_length = CreateMetalogDistribution(
# #   dataframe = iris,
# #   column_name = "Petal.Length"
# # )
# # metalog_petal_length = CreateMetalogDistribution(
# #   dataframe = iris,
# #   column_name = "Petal.Length",
# #   lower_boundary = 1
# # )
# # metalog_petal_length = CreateMetalogDistribution(
# #   dataframe = iris,
# #   column_name = "Petal.Length",
# #   upper_boundary = 10
# # )
# metalog_petal_length = CreateMetalogDistribution(
#   dataframe = iris,
#   column_name = "Petal.Length",
#   lower_boundary = 1,
#   upper_boundary = 10
# )
