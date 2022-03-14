# Load packages
library(missRanger)

# Declare function
ImputeMissingValues = function(dataframe,
                               number_of_trees = 1000,
                               random_seed = 412) {
  # Create imputed dataframe
  dataframe_imputed = missRanger(
    dataframe, 
    formula = . ~ . ,
    num.trees = number_of_trees, 
    verbose = 1, 
    seed = random_seed
  )
  
  # Return dataframe
  return(dataframe_imputed)
}

# Set arguments
data("airquality")
df_imputed = ImputeMissingValues(dataframe = airquality)
