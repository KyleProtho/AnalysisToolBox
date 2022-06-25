# Load packages
library(ggplot2)
library(dplyr)

# Declare function
SimulateDistributionOfProportion = function(sample_proportion,
                                            sample_size,
                                            confidence_interval = 0.95,
                                            outcome_variable_name = "Proportion",
                                            number_of_trials = 10000,
                                            random_seed = NULL,
                                            show_plot = TRUE,
                                            fill_color = "#4f92ff",
                                            return_simulated_proportions = TRUE) {
  # Calculate standard error
  standard_error = (sample_proportion * (1-sample_proportion)) / sample_size
  standard_error = sqrt(standard_error)
  
  # Calculate 95% confidence interval
  zscore = qnorm((1-confidence_interval)/2,
                 lower.tail=FALSE)
  
  # Print confidence interval
  lower_bound = round(sample_proportion - (zscore * standard_error), 3)
  upper_bound = round(sample_proportion + (zscore * standard_error), 3)
  print(paste("95% confidence interval lower bound:", lower_bound))
  print(paste("95% confidence interval upper bound:", upper_bound))
  
  # Set random seed if specified
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  
  # Conduct simulation and convert to df_sim_results
  list_sim_results = rnorm(n = number_of_trials,
                           mean = sample_proportion,
                           sd = standard_error)
  df_sim_results = as.data.frame(list_sim_results)
  
  # Add trial and variable columns, then drop list_sim_results column
  df_sim_results[["trial"]] = row.names(df_sim_results)
  df_sim_results[[outcome_variable_name]] = df_sim_results[["list_sim_results"]]
  df_sim_results$list_sim_results = NULL
  
  # Draw plot if requested
  if (show_plot) {
    p = ggplot(data=df_sim_results, 
               aes(x=.data[[outcome_variable_name]])) + 
      geom_histogram(fill=fill_color,
                     color="white",
                     alpha=0.8,
                     bins=20) +
      labs(title="Distribution",
           subtitle=outcome_variable_name,
           y="Count") + 
      theme_minimal() + 
      theme(axis.line=element_line(size=1, colour="#d6d6d6"),
            axis.text=element_text(color="#3b3b3b"),
            axis.text.x=element_text(size=12),
            panel.grid.major=element_blank(),
            panel.grid.minor=element_blank(),
            axis.title.x=element_blank()
      )
    plot(p)
  }
  
  # Return results
  if (return_simulated_proportions) {
    return(df_sim_results)
  }
}
