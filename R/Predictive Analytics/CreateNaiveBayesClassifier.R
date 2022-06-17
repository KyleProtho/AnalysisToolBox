# Load packages
library(caret)
library(caTools)
library(dplyr)
library(naivebayes)

# Declare function
CreateNaiveBayesClassifier = function(dataframe,
                                      outcome_variable,
                                      list_of_predictor_variables = NULL,
                                      share_of_training_set = 0.8,
                                      show_plot_of_model = TRUE,
                                      show_help = FALSE,
                                      random_seed = 412) {
  
  # If list of predictors is null, use all columns
  if (is.null(list_of_predictor_variables)) {
    list_of_predictor_variables = colnames(dataframe)
    list_of_predictor_variables = list_of_predictor_variables[list_of_predictor_variables != outcome_variable]
  }
  
  # Select predictors and outcome variable
  dataframe = dataframe %>% select(
    outcome_variable,
    all_of(list_of_predictor_variables)
  )
  
  # Set random seed
  set.seed(random_seed)
  
  # Split data into train and test data
  split = sample.split(dataframe, 
                       SplitRatio = share_of_training_set)
  dataframe_training = subset(dataframe, split == "TRUE")
  dataframe_test = subset(dataframe, split == "FALSE")
  
  # Create formula string
  if (is.null(list_of_predictor_variables)) {
    formula_string = paste(outcome_variable, "~ .")
  } else {
    formula_string = paste(outcome_variable, "~")
    for (i in 1:length(list_of_predictor_variables)) {
      if (i == 1) {
        formula_string = paste(formula_string, list_of_predictor_variables[i])
      } else {
        formula_string = paste(formula_string, "+", list_of_predictor_variables[i])
      }
    }
  }
  formula_string = as.formula(formula_string)
  
  # Create Naive Bayes Model to training dataset
  model = naive_bayes(formula_string,
                      data = dataframe_training)
  
  # If request, show plot of model
  if (show_plot_of_model) {
    plot(model)
  }
  
  # Test model and show confusion matrix
  if (share_of_training_set < 1) {
    # Predict on test data'
    y_pred = predict(model, 
                     newdata = dataframe_test)
    # Create and print confusion matrix
    cm = table(dataframe_test[[outcome_variable]], y_pred)
    confusionMatrix(cm)
  }
  
  # Create list (dictionary) to return
  return_dictionary = list(model,
                           dataframe_training,
                           dataframe_test)
  
  # If requested, print directions to access returned items
  if (show_help == TRUE) {
    writeLines(c("This function returns three items:",
                 "\t- Access the fitted model at the [[1]] index (you can use the 'summary()' method on this item)",
                 "\t- Access the training dataset at the [[2]] index",
                 "\t- Access the test dataset at the [[3]] index"),
               sep = "\n")
  }
  
  # Return model and train/test datasets
  return(return_dictionary)
}
