
# Load packages
library(arules)
library(arulesViz)
library(dplyr)
library(stringr)
library(tidyr)

# Declare function
ConductAssociationLearning = function(dataframe,
                                      key_column,
                                      items_column,
                                      support_threshold = NULL,
                                      confidence_threshold = 0.50,
                                      plot_association_rules = TRUE,
                                      plot_support = FALSE,
                                      random_seed = 412) {
  # Set seed, if specified
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  
  # If support_threshold not specified, set it to 2/unique count of key_column
  if (is.null(support_threshold)) {
    support_threshold = 2 / n_distinct(dataframe[[key_column]])
  }
  
  # Transform dataframe to wide form
  dataframe = dataframe %>% 
    select(
      !!sym(key_column),
      !!sym(items_column)
    ) %>%
    # Add TRUE
    mutate(const = TRUE) 
  
  # Convert to wide form
  dataframe = reshape(data = dataframe,
                      idvar = key_column,
                      timevar = items_column,
                      direction = "wide")
  colnames(dataframe) = str_replace(string = colnames(dataframe),
                                    pattern = "const.",
                                    replacement = "")
  
  # Convert to matrix, without the key column
  dataframe_matrix = as.matrix(dataframe[,-1])
  dataframe_matrix[is.na(dataframe_matrix)] = FALSE
  
  # Convert to transaction data type
  transaction_dataset = as(dataframe_matrix, "transactions")
  
  # Get association rules that are above thresholds
  assoc_rules = apriori(
    data = transaction_dataset,
    parameter = list(support = support_threshold,
                     confidence = confidence_threshold)
  )
  
  # Create association rule dataframe
  df_assoc_rules = as.data.frame(inspect(assoc_rules))
  df_assoc_rules$rule_name = paste(df_assoc_rules[["lhs"]],
                                   df_assoc_rules[,2],
                                   df_assoc_rules[["rhs"]])
  df_assoc_rules = df_assoc_rules %>% select(
    rule_name,
    lhs,
    rhs,
    count,
    support,
    confidence,
    lift,
    coverage
  ) %>% arrange(
    -count
  )
  
  # Show item frequency plot
  if (plot_support) {
    itemFrequencyPlot(transaction_dataset, topN = 10)
  }
  
  # Show association rule results
  inspect(sort(assoc_rules, by = 'lift')[1:10])
  p = plot(assoc_rules, 
           method = "graph",
           measure = "confidence",
           shading = "lift")
  plot(p)
  
  # Return association rules dataframe
  return(df_assoc_rules)
}

# # Example
# dataframe = read.csv("C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/CSV/2022 NFL Draft Picks.csv")
# table_assoc_rules = ConductAssociationLearning(dataframe = dataframe,
#                                                key_column = "Team",
#                                                items_column = "Position")
