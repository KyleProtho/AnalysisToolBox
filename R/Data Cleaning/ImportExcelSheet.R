# Load packages
library(readxl)
library(janitor)

# Declare function
ImportExcelSheet = function(filepath_to_excel_file,
                            sheet_index = 1,
                            clean_column_names = TRUE) {
  # Import data
  dataframe = read_excel(
    path = filepath_to_excel_file,
    sheet = sheet_index
  )
  
  # Replace blanks with NA
  for (variable in colnames(dataframe)) {
    dataframe[[variable]] == ifelse(test = dataframe[[variable]] == "",
                                    yes = NA,
                                    no = dataframe[[variable]])
    dataframe[[variable]] == ifelse(test = dataframe[[variable]] == " ",
                                    yes = NA,
                                    no = dataframe[[variable]])
  }
  
  # Clean column names if requested
  if (clean_column_names) {
    dataframe = dataframe %>% clean_names()
  }
  
  # Return dataframe
  return(dataframe)
}
