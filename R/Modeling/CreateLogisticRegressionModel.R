# Load packages
library(dplyr)
library(caTools)
library(stringr)
library(ggplot2)

# Declare function
CreateLogisticRegressionModel = function(dataframe,
                                         outcome_variable,
                                         list_of_predictors,
                                         share_for_training=.80,
                                         show_residual_plot=TRUE,
                                         show_help=TRUE,
                                         random_seed=NULL) {
  # Select columns to use in model
  list_of_variables = append(x = list_of_predictors,
                             values = outcome_variable)
  df_modeling = dataframe %>% 
    select(list_of_variables)
  
  # Keep complete cases only
  df_modeling = na.omit(df_modeling)
  
  # Print the sample size used for modeling
  print(paste("Number of observations eligible for modeling:", nrow(df_modeling)))
  
  # Ensure that the outcome variable is categorical
  df_modeling[[outcome_variable]] = as.factor(df_modeling[[outcome_variable]])
  
  # Set random seed for random selection of train-test split, if specified
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  
  # Split dataset into training and test set
  sampleSplit = sample.split(Y = df_modeling[[outcome_variable]],
                             SplitRatio = share_for_training)
  trainSet = subset(x=df_modeling, sampleSplit==TRUE)
  testSet = subset(x=df_modeling, sampleSplit==FALSE)
  
  # Print the sample size used for model training
  print(paste("Number of observations included in the training dataset:", nrow(trainSet)))
  
  # Generate formula string
  formula_string = paste(outcome_variable, "~", list_of_predictors[1])
  if (length(list_of_predictors) > 1) {
    for (i in 2:length(list_of_predictors)) {
      formula_string = paste(formula_string, "+", list_of_predictors[i])
      
    }
  }
  formula_string = as.formula(formula_string)
  
  # Create logistic regression model
  model = glm(formula_string, 
              data = trainSet,
              family = "binomial")
  
  # Show residual plot if requested
  df_residulals = as.data.frame(residuals(model))
  p = ggplot(data = df_residulals, 
             aes(x=residuals(model))) + 
    geom_histogram(fill="#2f7fff",
                   color="white",
                   alpha=0.70,
                   bins = 20) +
    labs(title="Residual plot",
         subtitle=str_wrap("This plot shows the distribution of errors the model has in predicting the training set. The residuals should resemble a normal distribution, although there is freedom for judgement.", width = 120),
         y="Count") + 
    theme_minimal() + 
    theme(axis.line=element_line(size=1, colour="#d6d6d6"),
          axis.text=element_text(color="#3b3b3b"),
          axis.text.x=element_text(size=12),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          axis.title.x=element_blank())
  plot(p)
  
  # Create list (dictionary) to return
  return_dictionary = list(model,
                           trainSet,
                           testSet)
  
  # If requested, print directions to access returned items
  if (show_help == TRUE) {
    writeLines(c("This function returns three items:",
                 "\t- Access the fitted model at the [[1]] index (you can use the 'summary()' method on this item)",
                 "\t- Access the training dataset at the [[2]] index",
                 "\t- Access the test dataset at the [[3]] index"),
               sep = "\n")
  }
  
  # Return items
  return(return_dictionary)
}
