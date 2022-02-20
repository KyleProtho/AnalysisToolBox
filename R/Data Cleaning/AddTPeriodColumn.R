# Load packages
library(lubridate)

# Define function
AddTPeriodColumn = function(dataframe,
                            date_column_name,
                            time_series_interval="days") {
  # Get earliest date
  earliest_date = min(dataframe[[date_column_name]])
  
  # Calculate difference from earliest date in interval
  if (time_series_interval=="seconds") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% seconds(1)
  } else if (time_series_interval=="minutes") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% minutes(1)
  } else if (time_series_interval=="hours") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% hours(1)
  } else if (time_series_interval=="days") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% days(1)
  } else if (time_series_interval=="weeks") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% weeks(1)
  } else if (time_series_interval=="months") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% months(1)
  } else if (time_series_interval=="years") {
    dataframe[["t_period"]] = interval(start=earliest_date,
                                       end=dataframe[[date_column_name]])
    dataframe[["t_period"]] = dataframe[["t_period"]] %/% months(1)
  } else {
    print("Not a valid interval argument. Valid argument is second, minutes, hours, days, weeks, months, or years.")
  }
  
  # Return updated dataframe
  return(dataframe)
}

