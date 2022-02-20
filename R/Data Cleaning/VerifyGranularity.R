# Load packages
library(dplyr)

# Define function
VerifyGranularity = function(dataframe,
                              list_of_key_columns,
                              add_dataset_key_to_dataframe=TRUE,
                              set_dataset_key_as_row_name=TRUE) {
  # Show number of rows in dataframe
  print(paste("Number of rows in dataframe:", nrow(dataframe)))
  
  # Iterate through list of key columnc and create key column
  for (i in 1:length(list_of_key_columns)) {
    if (i == 1) {
      dataframe[["dataset_key"]] = dataframe[[list_of_key_columns[i]]]
    } else {
      dataframe[["dataset_key"]] = paste(dataframe[["dataset_key"]], 
                                         "--",
                                         dataframe[[list_of_key_columns[i]]])
    }
  }
  
  # Show distinct count of key
  print(paste("Distinct count of dataset keys:", n_distinct(dataframe[["dataset_key"]])))
  
  # Delete dataset_key if user does not want to keep it
  if (add_dataset_key_to_dataframe == FALSE) {
    dataframe[["dataset_key"]] = NULL
  } else {
    if (set_dataset_key_as_row_name == TRUE) {
      row.names(dataframe) = dataframe[["dataset_key"]]
      dataframe[["dataset_key"]] = NULL
    } else {
      # Move dataset key to front of dataframe
      dataframe = select(dataframe, 
                         dataset_key, 
                         everything())
    }
  }
  
  # Return dataframe
  return(dataframe)
}
