# Load packages
library(zoo)

# Declare function
AddRollingAverageColumn = function(dataframe,
                                   value_column,
                                   rolling_average_length = 7,
                                   time_units = "days") {
  # Create new column name
  new_column_name = paste0(value_column, ".rolling.avg.", rolling_average_length, ".", time_units)
  
  # Add moving/rolling average
  dataframe[[new_column_name]] = rollmean(x = dataframe[[value_column]],
                                          k = rolling_average_length,
                                          fill = NA,
                                          align = "right")
  # Return dataframe
  return(dataframe)
}
