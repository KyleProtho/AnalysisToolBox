# Load packages

# Declare function
ConvertTimeSeriesToDataframe = function(time_series,
                                        value_name = "Value") {
  # Convert time series object to dataframe
  dataframe = data.frame(Time = time(time_series),
                         value_name = as.matrix(time_series))
  
  # Change column name of value
  dataframe[[value_name]] = dataframe$value_name
  dataframe$value_name = NULL
  
  # Return dataframe
  return(dataframe)
}
