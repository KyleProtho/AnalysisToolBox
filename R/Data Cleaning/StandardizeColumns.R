# Load packages

# Declare function
StandardizeColumns <- function(dataframe,
                               list_of_columns,
                               add_as_new_column = TRUE) {
  # Iterate through list of the columns
  for (i in 1:length(list_of_columns)) {
    # Create new column name
    if (add_as_new_column == TRUE) {
      new_column_name = paste0(list_of_columns[i], ".standardized")
    } else {
      new_column_name = list_of_columns[i]
    }
    
    # Standardize column
    dataframe[[new_column_name]] = scale(dataframe[[list_of_columns[i]]])[,1]
  }
  
  # Return updated dataframe
  return(dataframe)
}

# Test
data(iris)
iris = StandardizeColumns(dataframe = iris,
                          list_of_columns = c("Sepal.Length", "Sepal.Width"))
