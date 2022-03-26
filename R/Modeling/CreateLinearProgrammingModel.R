
# Load packages
library(lpSolve)
library(readxl)
library(janitor)
library(dplyr)
library(stringr)
library(tidyr)

# Set arguments
optimization_setup_excel = "C:/Users/oneno/OneDrive/Creations/Analysis Tool Box/Templates/Linear Optimization Model.xlsx"
all_constaints_integers = TRUE
show_help = TRUE

# Import decision variables
decision_var_setup = read_excel(path = optimization_setup_excel,
                                sheet = 1) %>%
  row_to_names(row_number = 1) %>%
  filter(
    !is.na(`Decision Variables`)
  )

# Convert decision variable attribute to numeric
for (variable in colnames(decision_var_setup[,4:ncol(decision_var_setup)])) {
  decision_var_setup[[variable]] = as.numeric(decision_var_setup[[variable]])
}
rm(variable)

# Get objective
objective_function = decision_var_setup$Objective[1]
objective_function = ifelse(
  test = objective_function == "Maximize",
  yes = "max",
  no = "min"
)

# Import constraints
constraint_setup = read_excel(path = optimization_setup_excel,
                              sheet = 2)

# Derive constraint operators
constraint_setup = constraint_setup %>% mutate(
  constraint_operator = case_when(
    str_detect(string = `Constraint Setting`, pattern = "Less than or equal to") ~ "<=",
    str_detect(string = `Constraint Setting`, pattern = "Greater than or equal to") ~ ">=",
    TRUE ~ "=="
  )
)

# Add non-negativity constraints, if there is one
is_non_negative = decision_var_setup$`Must be non-negative`[1]
if (is_non_negative == "Yes") {
  for (variable in c(decision_var_setup$`Decision Variables`)) {
    constraint_setup[nrow(constraint_setup) + 1,] = list(
      paste("Non-negativity of", variable),  # Constraint Name
      colnames(decision_var_setup)[4],  # Select Input/Column
      variable,  # Select Applicable Decision Variables
      "Greater than or equal to a value",  # Constraint Setting
      0,  # Constraint Amount
      ">="  # constraint_operator
    )
  }
}
rm(is_non_negative, variable)

# Add 'to-each' constraints, if there is one
constaints_to_each = constraint_setup %>%
  filter(
    `Select Applicable Decision Variables` == "To each"
  )
constraint_setup = constraint_setup %>%
  filter(
    `Select Applicable Decision Variables` != "To each"
  )
if (nrow(constaints_to_each) >= 1) {
  for (i in 1:nrow(constaints_to_each)) {
    for (variable in c(decision_var_setup$`Decision Variables`)) {
      constraint_setup[nrow(constraint_setup) + 1,] = list(
        paste(constaints_to_each$`Constraint Name`[i], "of", variable),  # Constraint Name
        colnames(decision_var_setup)[4],  # Select Input/Column
        variable,  # Select Applicable Decision Variables
        constaints_to_each$`Constraint Setting`[i],  # Constraint Setting
        constaints_to_each$`Constraint Amount`[i],  # Constraint Amount
        constaints_to_each$constraint_operator[i] # constraint_operator
      )
    }
  }
}
rm(constaints_to_each, i, variable)

# Convert "across all" constraints to dataframe
constraints_all = decision_var_setup %>%
  select(
    -one_of(c("Objective", "Must be non-negative")),
  )
constraints_all = as.data.frame(t(constraints_all)) %>%
  row_to_names(row_number = 1)
constraints_all[["Select Input/Column"]] = row.names(constraints_all)
constraints_all = constraints_all[-1,]
constraint_setup_all = constraint_setup %>%
  filter(
    `Select Applicable Decision Variables` == "Across all"
  ) %>%
  select(
    one_of("Select Input/Column", "Constraint Amount", "constraint_operator")
  )
constraints_all = left_join(
  constraints_all,
  constraint_setup_all,
  by = c("Select Input/Column")
) %>%
  select(
    "Select Input/Column",
    "Constraint Amount",
    "constraint_operator",
    everything()
  ) %>%
  filter(
    !is.na(`Constraint Amount`)
  )
for (variable in colnames(constraints_all[,4:ncol(constraints_all)])) {
  constraints_all[[variable]] = as.numeric(constraints_all[[variable]])
}
rm(variable, constraint_setup_all)

# Convert decision variable-specific constraints to dataframe
constraints_specific = data.frame()
for (i in 1:length(c(decision_var_setup$`Decision Variables`))) {
  variable = c(decision_var_setup$`Decision Variables`)[i]
  current_desc_var = constraint_setup %>% 
    filter(
      `Select Applicable Decision Variables` == variable
    ) %>%
    select(
      -one_of(c("Constraint Setting")),
    ) 
  current_desc_var[["Coef"]] = 1
  constraints_specific = bind_rows(
    constraints_specific,
    current_desc_var
  )
}
constraints_specific = constraints_specific %>% spread(
  key = "Select Applicable Decision Variables",
  value = "Coef"
)
constraints_specific[is.na(constraints_specific)] = 0
constraints_specific = constraints_specific %>%
  select(
    -one_of("Constraint Name")
  )
rm(i, variable, current_desc_var)

# TO-DO: Add constraints for column-based values


# Append constraints together
constraints = bind_rows(
  constraints_all,
  constraints_specific
)
rm(constraints_all, constraints_specific)

# Convert decision variable coefficients to matrix
constraints_coef = as.matrix(constraints[,4:ncol(constraints)])
decision_vars_coef = as.matrix(decision_var_setup[, 4])

# Create linear optimization model
optimization_model = lp(
  direction = objective_function,
  objective.in = decision_vars_coef,
  const.mat = constraints_coef,
  const.dir = c(constraints$constraint_operator),
  const.rhs = c(constraints$`Constraint Amount`),
  compute.sens = TRUE,
  all.int = all_constaints_integers  
)
rm(decision_vars_coef, constraints_coef)

# Add solution and outcome boundary for which the solution to works to decision_vars
decision_var_setup$solution = optimization_model$solution
decision_var_setup$solution_outcome_min = optimization_model$sens.coef.from
decision_var_setup$solution_outcome_max = optimization_model$sens.coef.to

# Add shadow price and its boundaries to decision_vars
decision_var_setup$shadow_price = tail(optimization_model$duals, nrow(decision_var_setup))
decision_var_setup$shadow_price_min = tail(optimization_model$duals.from, nrow(decision_var_setup))
decision_var_setup$shadow_price_max = tail(optimization_model$duals.to, nrow(decision_var_setup))

# Add shadow price and its boundaries to constraints
constraints$shadow_price = head(optimization_model$duals, nrow(constraints))
constraints$shadow_price_min = head(optimization_model$duals.from, nrow(constraints))
constraints$shadow_price_max = head(optimization_model$duals.to, nrow(constraints))
