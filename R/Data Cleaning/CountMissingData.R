
# Define function
CountMissingData <- function(dataframe) {
  # Iterate through columns and count null values
  for (variable in colnames(dataframe)) {
    missing_values = sum(is.na(dataframe[[variable]]))
    print(paste0("Number of missing values in ", variable, ": ", missing_values))
  }
}
