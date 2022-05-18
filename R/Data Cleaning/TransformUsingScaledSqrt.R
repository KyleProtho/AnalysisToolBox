# Load packages

# Declare function
TransformUsingScaledSqrt = function(dataframe,
                                    list_of_columns,
                                    add_as_new_column = TRUE) {
  
  # Iterate through list of columns
  for (variable in list_of_columns) {
    # Create new column name
    if (add_as_new_column) {
      new_column_name = paste0(variable, ".Scaled.sqrt")
    } else {
      new_column_name = variable
    }
    
    # Calculate scaled sqaured root
    dataframe[[new_column_name]] = sqrt(max(dataframe[[variable]] + 1) - dataframe[[variable]])
    
    # Set label
    new_column_label = str_replace_all(string = variable,
                                       pattern = "\\.",
                                       replacement = " ")
    new_column_label = str_replace_all(string = new_column_label,
                                       pattern = "-",
                                       replacement = " ")
    label(dataframe[[new_column_name]]) = paste("Scaled square root -", new_column_label)
  }
  
  # Return dataframe
  return(dataframe)
}
