# Load packages
library(naniar)
library(ggplot2)

# Define function
CountMissingData = function(dataframe,
                            show_incomplete_case_plot = TRUE,
                            show_missing_count_by_variable_plot = TRUE) {
  # If request, show plot of incomplete cases
  if (show_incomplete_case_plot) {
    p = gg_miss_case(dataframe,
                     show_pct = TRUE) + 
      labs(x = "Number of Cases") +
      theme_minimal()
    plot(p)
  }
  
  # Iterate through columns and count null values
  for (variable in colnames(dataframe)) {
    missing_values = sum(is.na(dataframe[[variable]]))
    print(paste0("Number of missing values in ", variable, ": ", missing_values))
  }
  
  # If requested, show plot
  if (show_missing_count_by_variable_plot) {
    p = gg_miss_var(dataframe,
                    show_pct = TRUE) +
      theme_minimal()
    plot(p)
  }
}
