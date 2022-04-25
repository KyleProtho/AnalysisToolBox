# Load packages
library(rmetalog)

# Declare function
SimulateOutcomeFromMetalog = function(metalog_distribution,
                                      number_of_terms = 4,
                                      number_of_trials = 10000,
                                      name_of_variable = NULL) {
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
