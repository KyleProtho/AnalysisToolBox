# Load packages
library(lpSolve)
library(readxl)
library(janitor)
library(dplyr)
# Note: This function is designed to work with the optimization model set-up Excel file: "C:\Users\oneno\OneDrive\Creations\Analysis Tool Box\Templates\Linear Optimization Model.xlsx"

# Declare function
CreateLinearOptimizationModel = function(optimization_setup_excel,
                                         all_constaints_integers = FALSE,
                                         show_help = TRUE) {
  # Import objective
  objective_function = read_excel(path = optimization_setup_excel,
                                  sheet = 1)
  objective_function = colnames(objective_function)[2]
  objective_function = ifelse(
    test = objective_function == "Maximize",
    yes = "max",
    no = "min"
  )
  
  # Import decision variables
  decision_vars = read_excel(path = optimization_setup_excel,
                             sheet = 2) %>%
    clean_names()
  
  # Import constraints
  constraints = read_excel(path = optimization_setup_excel,
                           sheet = 3) %>%
    row_to_names(row_number = 1) %>%
    clean_names() %>%
    mutate(
      constraint_rule = case_when(
        constraint_setting == "Less than or equal to" ~ "<=",
        constraint_setting == "Greater than or equal to" ~ ">=",
        TRUE ~ "=="
      )
    )
  
  # Convert decision var's coefficients to matrix
  decision_vars_coef = constraints[, 5:ncol(constraints)-1]
  decision_vars_coef = as.matrix(decision_vars_coef)
  
  # Create linear optimization model
  optimization_model = lp(
    direction = objective_function, 
    objective.in = c(decision_vars$outcome), 
    const.mat = decision_vars_coef, 
    const.dir = c(constraints$constraint_rule),
    const.rhs = c(constraints$constraint_amount),
    compute.sens = TRUE,
    all.int = all_constaints_integers
  )
  
  # Add solution and outcome boundary for which the solution to works to decision_vars
  decision_vars$solution = optimization_model$solution
  decision_vars$solution_outcome_min = optimization_model$sens.coef.from
  decision_vars$solution_outcome_max = optimization_model$sens.coef.to
  
  # Add shadow price and its boundaries to decision_vars
  decision_vars$shadow_price = tail(optimization_model$duals, nrow(decision_vars))
  decision_vars$shadow_price_min = tail(optimization_model$duals.from, nrow(decision_vars))
  decision_vars$shadow_price_max = tail(optimization_model$duals.to, nrow(decision_vars))
  
  # Add shadow price and its boundaries to constraints
  constraints$shadow_price = head(optimization_model$duals, nrow(constraints))
  constraints$shadow_price_min = head(optimization_model$duals.from, nrow(constraints))
  constraints$shadow_price_max = head(optimization_model$duals.to, nrow(constraints))
  
  # Print results
  if (optimization_model$status == 2) {
    print("No feasible solution found -- meaning it is not possible to satisfy all constraints in this problem.")
  } else if (optimization_model$status == 3) {
    if (objective_function == "max") {
      print("The solution to this linear optimization problem is unbounded (it is infinitely large).")
    } else {
      print("The solution to this linear optimization problem is unbounded (it is infinitely small).")
    }
  } else {
    objective = ifelse(
      test = objective_function == "max",
      yes = "maximize",
      no = "minimize"
    )
    print(paste("Below shows the optimal decisions to", objective, "the outcome."))
    for (i in 1:length(optimization_model$solution)) {
      print(paste0(decision_vars$decision_variables[i], ": ", optimization_model$solution[i]))
    }
  }
  
  # If requested, print helper text
  if (show_help == TRUE) {
    writeLines(c("\nThis function returns three items:",
                 "\t- Access the linear optimization model at the [[1]] index",
                 "\t- Access the decision variables dataset with solutions and shadow prices at the [[2]] index",
                 "\t- Access the constraints dataset with shadow prices at the [[3]] index"),
               sep = "\n")
  }
  
  # Return results
  return_list = list(optimization_model,
                     decision_vars,
                     constraints)
  return(return_list)
}

# Test
results = CreateLinearOptimizationModel(optimization_setup_excel = "C:/Users/oneno/OneDrive/Creations/Analysis Tool Box/Templates/Linear Optimization Model.xlsx",
                                        all_constaints_integers = TRUE)
