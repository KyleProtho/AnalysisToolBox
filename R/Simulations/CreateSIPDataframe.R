# Load packages
library(dplyr)

# Declare function
CreateSIPDataframe = function(name_of_items,
                              list_of_items,
                              number_of_trials=10000) {
  # Create place holder dataframe
  df_sip = data.frame()
  
  # Keep only unique set of items
  list_of_items = unique(list_of_items)
  
  # Iterate through list of items
  for (i in 1:length(list_of_items)) {
    # Create dataframe of rows for each trial
    df_temp = data.frame(1:number_of_trials)
    colnames(df_temp) = "Trial"
    
    # Add item as column
    df_temp[[name_of_items]] = list_of_items[i]
    
    # Concatenate item and trial number as key
    row.names(df_temp) = paste0(df_temp[["Trial"]], "-", df_temp[[name_of_items]])
    
    # Append to placeholder dataframe
    df_sip = bind_rows(df_sip, df_temp)
    rm(df_temp)
  }
  
  # Return dataframe
  return(df_sip)
}
