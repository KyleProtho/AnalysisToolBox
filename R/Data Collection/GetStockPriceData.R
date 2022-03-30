# Load packages
library(tidyquant)
library(lubridate)
library(stringr)

# Declare function
GetStockPriceData = function(list_of_tickers,
                             to_date = lubridate::today(),
                             from_date = lubridate::today() - 365) {
  # Fetch price data for tickers
  dataframe = tq_get(list_of_tickers,
                     from = from_date,
                     to = to_date,
                     get = "stock.prices")
  
  # Change column names to title case
  colnames(dataframe) = str_to_title(colnames(dataframe))
  
  # Return data
  return(dataframe)
}
