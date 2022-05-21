# Load packages
library(dplyr)
library(factoextra)
library(nFactors)
library(psych)

# Declare function
ConductFactorAnalysis = function(dataframe,
                                 list_of_variables = NULL,
                                 number_of_factors_to_create = NULL,
                                 measure_of_sampling_adequacy_threshold = 0.5,
                                 pca_standard_deviation_threshold = 1,
                                 standardize_variables = TRUE,
                                 plot_pca_results = TRUE) {
  # Select variables and keep complete cases only
  if (is.null(list_of_variables)) {
    dataframe_kmo = dataframe %>% 
      dplyr::select_if(is.numeric) %>% 
      na.omit
  } else {
    dataframe_kmo = dataframe %>% 
      select(
        all_of(list_of_variables)
      ) %>%
      na.omit
  }
  
  # Run KMO test
  kmo_results = KMO(dataframe_kmo)
  print(kmo_results)
  message("The KMO (Kaiser-Meyer-Olkin) test aims to estimate the proportion of variance among the variables you selected that may be caused by some underlying factor(s).")
  message("The closer the Overall Measure of Sampling Adequacy (MSA) is to 1, the more useful factor analysis will be for your selected variables.")
  
  # See if KMO test result is below threshold 
  if (kmo_results$MSA < measure_of_sampling_adequacy_threshold) {
    warning(paste0("The Overall MSA is below your threshold of ", measure_of_sampling_adequacy_threshold, ". Factor analysis is not recommended!"))
  } else {
    message(paste("The Overall MSA is above your threshold of", measure_of_sampling_adequacy_threshold))
  }
  
  # Get list of variables that are above MSA threshold
  list_of_pca_eligible_vars = as.data.frame(kmo_results$MSAi)
  names(list_of_pca_eligible_vars)[names(list_of_pca_eligible_vars) == "kmo_results$MSAi"] = "MSA"
  list_of_pca_eligible_vars = list_of_pca_eligible_vars %>%
    filter(
      MSA >= measure_of_sampling_adequacy_threshold
    )
  list_of_pca_eligible_vars = c(row.names(list_of_pca_eligible_vars))
  
  # Select eligible variables and keep complete cases only
  dataframe_pca = dataframe %>% 
    select(all_of(list_of_pca_eligible_vars)) %>% 
    na.omit
  
  # Conduct PCA analysis
  pca_model = princomp(dataframe_pca,
                       cor = standardize_variables,
                       scores = TRUE)
  
  # If requested, show graph of variables
  if (plot_pca_results) {
    p = fviz_pca_var(
      pca_model,
      col.var = "contrib",
      gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
      repel = TRUE
    )
    plot(p)
  }
  
  # Get number of factors to use based on st. dev., or eigenvalue, threshold
  pca_sdev = as.data.frame(pca_model$sdev)
  names(pca_sdev)[names(pca_sdev) == "pca_model$sdev"] = "sdev"
  pca_sdev = pca_sdev %>% 
    filter(
      sdev >= 1
    )
  
  # Conduct final factor analysis
  factor_analysis_results = factanal(dataframe_pca, 
                                     nrow(pca_sdev), 
                                     rotation="varimax")
  
  # Show factor analysis results
  print(factor_analysis_results, 
        digits=2, 
        cutoff=.3, 
        sort=TRUE)
  
  # Select only factor-relevant columns
  dataframe_factors = dataframe %>% 
    select(all_of(list_of_pca_eligible_vars))
  
  # Add factors to original dataset
  df_factors = factor.scores(dataframe_factors, factor_analysis_results)
  df_factors = as.data.frame(df_factors$scores)
  dataframe = bind_cols(
    dataframe,
    df_factors
  )
  
  # Return dataframe
  return(dataframe)
}
