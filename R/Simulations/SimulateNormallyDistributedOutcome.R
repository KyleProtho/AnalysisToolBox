# Load packages

# Declare function
SimulateNormallyDistributedOutcome = function(expected_outcome,
                                              standard_deviation,
                                              outcome_variable_name,
                                              number_of_trials = 10000,
                                              random_seed = NULL) {
  # Set random seed if specified
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  
  # Conduct simulation and convert to dataframe
  list_sim_results = rnorm(n = number_of_trials,
                           mean = expected_outcome,
                           sd = standard_deviation)
  df_sim_results = as.data.frame(list_sim_results)
  
  # Add trial and variable columns, then drop list_sim_results column
  df_sim_results[["trial"]] = row.names(df_sim_results)
  df_sim_results[[outcome_variable_name]] = df_sim_results[["list_sim_results"]]
  df_sim_results$list_sim_results = NULL
  
  # Return dataframe
  return(df_sim_results)
}

