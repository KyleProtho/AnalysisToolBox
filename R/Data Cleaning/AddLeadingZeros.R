# Load packages
library(numform)

# Define function
AddLeadingZeros <- function(dataframe,
                            column_name,
                            fixed_width=NULL,
                            add_as_new_column=FALSE) {
  # If fixed width argument is null, set it to the longest string in column
  if (is.null(fixed_width)) {
    fixed_width = max(nchar(as.character(dataframe[[column_name]])))
  }
  
  # Create new column name
  if (add_as_new_column == TRUE) {
    new_column_name = paste(column_name, ".leading_zeros")
  } else {
    new_column_name = column_name
  }
  
  # Add leading zeros to the left
  dataframe[[new_column_name]] = f_pad_zero(as.character(dataframe[[column_name]]),
                                            width = fixed_width)
  
  # Return updated dataframe
  return(dataframe)
  
}
