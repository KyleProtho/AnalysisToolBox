# Load packages
library(crypto2)
library(lubridate)
library(dplyr)
library(stringr)

# Declare function
GetCryptocurrencyPriceData = function(list_of_tickers,
                                      to_date = lubridate::today(),
                                      from_date = lubridate::today() - 365) {
  # Convert list of symbols to crypto2 format
  list_of_tickers_crypto2 = crypto_list(only_active=TRUE) %>%
    filter(
      symbol %in% list_of_tickers
    )
  
  # Convert date to appropriate format
  to_date = format(to_date, "%Y%m%d")
  from_date = format(from_date, "%Y%m%d")
  
  # Get price data
  dataframe = crypto_history(
    coin_list = list_of_tickers_crypto2,
    convert = "USD",
    start_date = from_date,
    end_date = to_date
  )
  
  # Convert column names to title case
  colnames(dataframe) = str_to_title(colnames(dataframe))
  
  # Return dataframe
  return(dataframe)
}
