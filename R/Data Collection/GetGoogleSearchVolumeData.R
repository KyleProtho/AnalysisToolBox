# Load packages
library(gtrendsR)
library(dplyr)
library(lubridate)

# Declare function
GetGoogleSearchVolumeData = function(list_of_terms,
                                     period = "today 12-m",
                                     google_product = "web",
                                     only_interest_over_time = TRUE,
                                     list_of_country_iso2c = c("US", "CA", "MX"),
                                     normalize_by_term = TRUE,
                                     geographies_to_return = c("country", "state", "metro_area", "city"),
                                     return_related_queries = TRUE) {
  
  # Fetch Google search volume
  df_trends = data.frame()
  if (normalize_by_term == TRUE) {
    for (variable in list_of_terms) {
      df_temp = gtrends(
        keyword = c(variable),
        geo = list_of_country_iso2c,
        time = period,
        gprop = google_product,
        compared_breakdown = FALSE,
        low_search_volume = TRUE,
        tz = 0,
        onlyInterest = only_interest_over_time
      )
      df_trends = bind_rows(
        df_trends,
        df_temp
      )
    }
  } else {
    for (variable in list_of_country_iso2c) {
      df_temp = gtrends(
        keyword = list_of_terms,
        geo = c(variable),
        time = period,
        gprop = google_product,
        compared_breakdown = TRUE,
        low_search_volume = TRUE,
        tz = 0,
        onlyInterest = only_interest_over_time
      )
      df_trends = bind_rows(
        df_trends,
        df_temp
      )
    }
  }
  
  # Get results by time
  df_trends_over_time = df_trends$interest_over_time
  
  # Get related queries
  df_related_queries = df_trends$related_queries
  
  # Create list of items to return
  list_return = list()
  list_return[["time_series"]] = df_trends_over_time
  
  if (only_interest_over_time == FALSE) {
    # Append related queries if requested
    if (return_related_queries) {
      list_return[["related_queries"]] = df_related_queries
    }
    
    # Get results by geography
    df_trends_by_country = df_trends$interest_by_country
    df_trends_by_state = df_trends$interest_by_region
    df_trends_by_dma = df_trends$interest_by_dma
    df_trends_by_city = df_trends$interest_by_city
    
    # Append to list of return items
    if ("country" %in% geographies_to_return) {
      list_return[["by_country"]] = df_trends_by_country
    }
    if ("state" %in% geographies_to_return) {
      list_return[["by_state"]] = df_trends_by_state
    }
    if ("metro_area" %in% geographies_to_return) {
      list_return[["by_metro_area"]] = df_trends_by_dma
    }
    if ("city" %in% geographies_to_return) {
      list_return[["by_city"]] = df_trends_by_city
    }
    return(list_return)
  } else {
    # Return results
    return(df_trends_over_time)
  }
}
