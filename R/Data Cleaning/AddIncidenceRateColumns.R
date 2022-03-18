# Load packages
library(table1)
library(stringr)

# Declare function
AddIncidenceRateColumns = function(dataframe,
                                   list_of_columns,
                                   population_column,
                                   per_n = 10000,
                                   per_n_label = "people") {
  # Iterate through list of columns
  for (variable in list_of_columns) {
    # Create new column name
    new_column_name = paste0(variable, ".per", per_n, ".people")
    
    # Calculate rate
    dataframe[[new_column_name]] = dataframe[[variable]] / dataframe[[population_column]] * per_n
    
    # Set label
    new_column_label = str_replace_all(string = variable,
                                       pattern = "\\.",
                                       replacement = " ")
    new_column_label = str_replace_all(string = new_column_label,
                                       pattern = "-",
                                       replacement = " ")
    new_column_label = paste(new_column_label, "per", format(as.integer(per_n), nsmall=1, big.mark=","), per_n_label)
    label(dataframe[[new_column_name]]) = new_column_label
  }
  
  # Return dataframe
  return(dataframe)
}