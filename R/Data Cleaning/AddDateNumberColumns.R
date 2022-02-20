# Load packages
library(lubridate)

# Define function
AddDateNumberColumns = function(dataframe,
                                date_column_name) {
  # Extract year number from date column
  dataframe[[paste0(date_column_name, ".year")]] = year(dataframe[[date_column_name]])
  
  # Extract month number from date column
  dataframe[[paste0(date_column_name, ".month")]] = month(dataframe[[date_column_name]])
  dataframe[[paste0(date_column_name, ".month_label")]] = month(dataframe[[date_column_name]],
                                                                label=TRUE)
  
  # Extract week number from date column
  dataframe[[paste0(date_column_name, ".week")]] = week(dataframe[[date_column_name]])
  
  # Extract day number from date column
  dataframe[[paste0(date_column_name, ".day_of_year")]] = yday(dataframe[[date_column_name]])
  
  # Extract day of month number from date column
  dataframe[[paste0(date_column_name, ".day_of_month")]] = mday(dataframe[[date_column_name]])
  
  # Extract day of week number from date column
  dataframe[[paste0(date_column_name, ".day_of_week")]] = wday(dataframe[[date_column_name]])
  dataframe[[paste0(date_column_name, ".day_of_week_label")]] = wday(dataframe[[date_column_name]],
                                                                     label=TRUE)
  
  # Return dataframe
  return(dataframe)
}
