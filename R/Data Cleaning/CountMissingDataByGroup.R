
# Load packages
library(dplyr)
library(stringr)

# Define function
CountMissingDataByGroup <- function(dataframe,
                                    list_of_grouping_variables) {
  # Group and summarize null count
  df_missing = dataframe %>% 
    group_by(across(all_of(list_of_grouping_variables))) %>% 
    summarize_all(.funs = funs('NA' = sum(is.na(.))))
  
  
  # Remove _NA from column names
  colnames(df_missing) = str_replace_all(string=colnames(df_missing),
                                         pattern="_NA",
                                         replacement="")
  
  # Return missing data summary
  return(df_missing)
}
