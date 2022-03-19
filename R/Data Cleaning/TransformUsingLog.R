# Load packages
library(table1)
library(stringr)

# Declare function
TransformUsingLog = function(dataframe,
                             list_of_columns,
                             add_as_new_column = TRUE) {
  # Iterate through list of columns
  for (variable in list_of_columns) {
    # Create new column name
    if (add_as_new_column) {
      new_column_name = paste0("Log.", variable)
    } else {
      new_column_name = variable
    }
    
    # Transform using log function
    dataframe[[new_column_name]] = log(dataframe[[variable]])
    
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