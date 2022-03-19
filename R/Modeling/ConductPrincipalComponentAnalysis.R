# Load packages
library(dplyr)
library(tidyr)
library(factoextra)
library(naniar)

# Declare function
ConductPrincipalComponentAnalysis = function(dataframe,
                                             list_of_key_columns,
                                             list_of_predictor_variables = NULL,
                                             missing_data_threshold = 0.20,
                                             show_graph_of_variables = TRUE,
                                             scale_pca = TRUE) {
  # Reduce dataframe to numerical variables only if list_of_predictor_variables not provided. Otherwise, select variables listed
  if (is.null(list_of_predictor_variables)) {
    dataframe_num = dataframe %>%
      select(
        -all_of(list_of_key_columns)
      ) %>% 
      select(
        where(is.numeric)
      )
  } else {
    dataframe_num = dataframe %>%
      select(
        all_of(list_of_predictor_variables)
      )
  }
  
  # Column bind key columns
  dataframe_keys = dataframe %>%
    select(
      all_of(list_of_key_columns)
    )
  dataframe = bind_cols(
    dataframe_keys,
    dataframe_num
  )
  rm(dataframe_keys, dataframe_num)
  
  # Convert to long form dataframe
  df_inclusion = dataframe %>% 
    gather(
      key = "Variable",
      value = "Value",
      -list_of_key_columns
    )
  
  # Group and summarize missing data by column
  df_inclusion = df_inclusion %>%
    group_by(Variable) %>% 
    miss_var_summary() %>%
    rename(
      Missing.Count = n_miss,
      Missing.Percent = pct_miss
    ) %>%
    summarize(
      Missing.Count = sum(Missing.Count),
      Missing.Percent = sum(Missing.Percent)
    ) %>% mutate(
      Missing.Percent = Missing.Percent / 100
    )
  
  # Add flag showing if variable if below threshold specified
  df_inclusion$Include = ifelse(
    test = df_inclusion$Missing.Percent <= missing_data_threshold,
    yes = TRUE,
    no = FALSE
  )
  
  # Select all qualifying columns
  df_inclusion = df_inclusion %>%
    filter(
      Include == TRUE
    )
  dataframe = dataframe %>%
    select(
      all_of(c(df_inclusion$Variable))
    )
  
  # Keep complete cases only
  dataframe = dataframe[complete.cases(dataframe), ]
  
  # Conduct PCA
  pca_model = prcomp(dataframe,
                     center = TRUE,
                     scale. = scale_pca)
  
  # If requested, show graph of variabbles
  if (show_graph_of_variables) {
    p = fviz_pca_var(
      pca_model,
      col.var = "contrib",
      gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
      repel = TRUE
    )
    plot(p)
  }
  
  # Return PCA model
  return(pca_model)
}

# Test
data("airquality")
ConductPrincipalComponentAnalysis(dataframe = airquality,
                                  list_of_key_columns = c("Month", "Day"))
