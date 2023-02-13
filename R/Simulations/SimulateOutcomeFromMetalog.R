# Load packages
library(rmetalog)

# Declare function
SimulateOutcomeFromMetalog = function(metalog_distribution,
                                      number_of_terms = 4,
                                      number_of_trials = 10000,
                                      name_of_variable = NULL) {
  
  # Ensure that number of terms and number of trails are integers
  number_of_terms = as.numeric(as.integer(number_of_terms))
  number_of_trials = as.numeric(as.integer(number_of_trials))
  
  # Generate SIP
  df_simulated = rmetalog(metalog_distribution, 
                          n = number_of_trials, 
                          term = number_of_terms)
  
  # If name of variable not specified, make one
  if (is.null(name_of_variable)) {
    name_of_variable = "simulation_result"
  }
  
  # Create dataframe
  df_simulated = data.frame(df_simulated)
  colnames(df_simulated) = name_of_variable
  
  # Return simulated values
  return(df_simulated)
}


# # Test the function
# iris = iris
# hist(iris[["Petal.Length"]], col = "red", main = "Petal length - Observed values")
# source("https://raw.githubusercontent.com/onenonlykpro/SnippetsForStatistics/master/R/Simulations/CreateMetalogDistribution.R")
# dist_petal_length = CreateMetalogDistribution(
#   dataframe=iris,
#   column_name="Petal.Length",
#   boundary_setting="b",
#   upper_boundary=8,
#   lower_boundary=1,
#   plot_fitted_distributions=FALSE
# )
# summary(dist_petal_length)
# sip_petal_length = SimulateOutcomeFromMetalog(
#   metalog_distribution=dist_petal_length,
#   number_of_terms=7,
#   number_of_trials=10000 
# )
# hist(sip_petal_length[["simulation_result"]], col = "blue", main = "Petal length - Simulated values")
