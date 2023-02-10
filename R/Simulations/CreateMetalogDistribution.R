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
# MyMetalog = CreateMetalogDistribution(dataframe = mtcars,
#                                    column_name = "mpg")

