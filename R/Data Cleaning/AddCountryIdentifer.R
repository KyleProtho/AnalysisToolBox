# Load packages
library(countrycode)

# Define function
AddCountryIdentifiers = function(dataframe,
                                 country_name_column) {
  # Add new columns for commonly used country identifiers
  dataframe[["ISO3C"]] = countrycode(sourcevar=dataframe[[country_name_column]],
                                     origin="country.name",
                                     destination = "iso3c")
  dataframe[["ISO2C"]] = countrycode(sourcevar=dataframe[[country_name_column]],
                                     origin="country.name",
                                     destination = "iso2c")
  dataframe[["World.Bank.Code"]] = countrycode(sourcevar=dataframe[[country_name_column]],
                                               origin="country.name",
                                               destination = "wb")
  dataframe[["IMF.Code"]] = countrycode(sourcevar=dataframe[[country_name_column]],
                                        origin="country.name",
                                        destination = "imf")
  # Return updated dataset
  return(dataframe)
}
