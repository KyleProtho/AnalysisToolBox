# Load packages
library(caret)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rsample)

# Declare function
CreateRegressionTree = function(dataframe,
                                outcome_variable,
                                list_of_predictor_variables = NULL,
                                share_for_training = .80,
                                use_bootstrap_aggregating = TRUE,
                                number_of_folds_for_cross_validation = 10,
                                show_important_predictor_plot = TRUE,
                                show_regression_tree_plot = TRUE,
                                show_help = TRUE,
                                random_seed = 412) {
  # Keep complete cases only
  if (is.null(list_of_predictor_variables)) {
    dataframe_complete = dataframe %>% 
      na.omit
  } else{
    dataframe_complete = dataframe %>% 
      select(all_of(outcome_variable, list_of_predictor_variables)) %>%
      na.omit
  }
  
  # Split into test and training datasets
  set.seed(random_seed)
  data_split = initial_split(dataframe_complete, prop = share_for_training)
  data_train = training(data_split)
  data_test = testing(data_split)
  
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
  
  # Create regression tree model
  if (use_bootstrap_aggregating) {
    ctrl = trainControl(method = "cv",  
                        number = number_of_folds_for_cross_validation)
    # Create bagged regression tree model
    regression_tree_model = train(
      formula_string,
      data = data_train,
      method = "treebag",
      trControl = ctrl,
      importance = TRUE
    )
    
    # Show plot of most important predictors
    if (show_important_predictor_plot) {
      V = varImp(regression_tree_model)[["importance"]]
      p = ggplot(V, 
                 aes(x=reorder(rownames(V), Overall), 
                     y=Overall)) +
        geom_point(color="blue", 
                   size=4, 
                   alpha=0.6) +
        geom_segment(aes(x=rownames(V), xend=rownames(V), y=0, yend=Overall), 
                     color='skyblue') +
        xlab('Variable') +
        ylab('Importance') +
        theme_minimal() +
        coord_flip() 
      plot(p)
    }
    
  } else {
    # Create regression tree model
    regression_tree_model = rpart(
      formula = formula_string,
      data = data_train,
      method = "anova"
    )
    
    # Show plot of regression tree model
    if (show_regression_tree_plot) {
      rpart.plot(regression_tree_model)
    }
  }
  
  # Create list (dictionary) to return
  return_dictionary = list(regression_tree_model,
                           data_train,
                           data_test)
  
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
?train
