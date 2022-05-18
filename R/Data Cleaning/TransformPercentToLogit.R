# Load packages

# Declare function
TransformPercentToLogit = function(dataframe,
                                   list_of_percent_columns,
                                   add_as_new_column = TRUE,
                                   keep_odds = FALSE) {
  # Iterate through list of columns
  for (variable in list_of_percent_columns) {
    # Create new column name
    if (add_as_new_column) {
      new_column_name = paste0(variable, ".Logit")
    } else {
      new_column_name = variable
    }
    
    # Convert percent to odds
    odds_column = paste0(variable, ".Odds")
    dataframe[[odds_column]] = dataframe[[variable]] / (1 - dataframe[[percent_column]])
    
    # Convert odds to logit
    logit_column = paste0(variable, ".Logit")
    dataframe[[logit_column]] = log(dataframe[[odds_column]])
    
    # Remove odds column
    if (keep_odds == FALSE) {
      dataframe[[odds_column]] = NULL
    }
    
    # Set label
    new_column_label = str_replace_all(string = variable,
                                       pattern = "\\.",
                                       replacement = " ")
    new_column_label = str_replace_all(string = new_column_label,
                                       pattern = "-",
                                       replacement = " ")
    label(dataframe[[new_column_name]]) = paste("Logit -", new_column_label)
  }
  
  # Return dataframe
  return(dataframe)
}