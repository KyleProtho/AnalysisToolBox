# Load packages
library(xlsx)

# Declare function
ImportExcelSheet = function(filepath_to_excel_file,
                            sheet_index = 1) {
  # Import data
  dataframe = read.xlsx(
    file = filepath_to_excel_file,
    sheetIndex = sheet_index
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
  
  # Return dataframe
  return(dataframe)
}
