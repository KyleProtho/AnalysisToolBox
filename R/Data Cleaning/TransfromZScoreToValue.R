# Load packages

# Declare function
TransformZScoreToValue = function(dataframe,
                                  zscore_column,
                                  mean,
                                  standard_deviation,
                                  name_of_value = NULL) {
  # Get name of value
  if (is.null(name_of_value)) {
    name_of_value = "Original value"
  }
  
  # Calculate original value
  dataframe[name_of_value] = (dataframe[zscore_column] * standard_deviation) + mean
  
  # Return updated dataframe
  return(dataframe)
}
