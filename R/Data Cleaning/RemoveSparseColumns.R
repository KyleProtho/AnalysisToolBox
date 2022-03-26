# Load packages
library(dplyr)

# Declare function
RemoveSparseColumns = function(dataframe,
                               list_columns_to_ignore,
                               missingness_threshold = .20) {
  # Preserve outcome columns
  df_cleaned = dataframe %>%
    select(
      one_of(list_columns_to_ignore)
    )
  dataframe = dataframe %>%
    select(
      -one_of(list_columns_to_ignore)
    )
  
  # Remove sparse columns
  dataframe = dataframe[, colSums(is.na(dataframe)) <= missingness_threshold * nrow(dataframe)]
  
  # Bind outcome columns back to dataset
  dataframe = bind_cols(
    dataframe,
    df_cleaned
  )
  
  # Return cleaned dataframe
  return(dataframe)
}