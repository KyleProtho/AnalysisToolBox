from CHAID import Tree
import graphviz as gv
from matplotlib import pyplot as plt
import os
import pandas as pd
import plotly
import plotly.graph_objects as go
from sklearn.tree import plot_tree
import statsmodels.api as sm

# Set arguments
dataframe = sm.datasets.get_rdataset("ResumeNames",
                                package = "AER").data 
outcome_variable = "call"
list_of_ordered_categorical_predictors = [
    'quality', 'jobs'
]
list_of_unordered_categorical_predictors = [
    'gender', 'ethnicity'
]
CHAID_tree_max_depth=3
CHAID_tree_minimum_node_size=30
CHAID_tree_split_threshold=0.05

# Select necessary columns
dataframe = dataframe[[outcome_variable] + list_of_ordered_categorical_predictors + list_of_unordered_categorical_predictors]

# Iterate through ordered categorical variables and convert to ordinal
if len(list_of_ordered_categorical_predictors) > 0:
    for predictor in list_of_ordered_categorical_predictors:
        # Sort by predictor
        dataframe = dataframe.sort_values(by=predictor)
        # Assign a number to each unique value
        dataframe[predictor] = pd.Categorical(dataframe[predictor],
                                              ordered=True).codes

# Iterate through unordered categorical variables and convert to nominal
if len(list_of_unordered_categorical_predictors) > 0:
    for predictor in list_of_unordered_categorical_predictors:
        # Sort by predictor
        dataframe = dataframe.sort_values(by=predictor)
        # Assign a number to each unique value
        dataframe[predictor] = pd.Categorical(dataframe[predictor],
                                              ordered=False).codes

# Convert outcome variable to binary
dataframe[outcome_variable] = pd.Categorical(dataframe[outcome_variable],
                                             ordered=False).codes

# Create dictionary of variable names and types
X_names = list_of_ordered_categorical_predictors + list_of_unordered_categorical_predictors
dict_predictor_variables = dict(zip(
    X_names,
    ['ordinal']*len(list_of_ordered_categorical_predictors) + ['nominal'] * len(list_of_unordered_categorical_predictors)
))
                                            
# Create CHAID tree
model = Tree.from_pandas_df(
    df=dataframe,
    i_variables=dict_predictor_variables,
    d_variable=outcome_variable,
    max_depth=CHAID_tree_max_depth,
    min_child_node_size=CHAID_tree_split_threshold,
    split_threshold=CHAID_tree_split_threshold
)

# Print population size
print("Population size:", model.data_size)
# Print tree summary
print("Number of levels:", model.max_depth)
# Print tree accuracy
print("Accuracy: " + str(round(model.accuracy(), 4) * 100) + "%")
print("\n")

# # Print CHAID tree
# model.print_tree()
# Iterate through nodes

for node in model:
    # print(node)
    # print(node.__dict__)
    # print(dir(node.split))
    # Set split column to "" if it is None
    try:
        split_column = str(node.split.column)
    except TypeError:
        split_column = ""
    
    # Get parent node ID and info
    parent_node_id = node.parent
    try:
        parent_node_info = model.get_node(parent_node_id)
        parent_node_var = parent_node_info.split.column
        parent_node_value = node.choices
        parent_node = str(parent_node_var + " " + str(parent_node_value)) 
    except TypeError:
        parent_node = "root"
        
    # Get p-value of split
    try:
        node_p_value = node.split.p
        node_p_value = round(node_p_value, 4)
    except TypeError:
        node_p_value = "N/A"
    
    # Get denominator of the split
    parent_node_info = model.get_node(0)
    denominator = parent_node_info.indices.shape[0]
    
    # Pretty print node information
    for group in node.split.split_groups:
        if parent_node == "root":
            continue
        else:
            # Get numerator of the split by adding the values of the .members attribute
            numerator = 0
            for key, value in node.members.items():
                numerator += value
            
            # Calculate percentage of split
            if denominator == 0:
                proportion = "N/A"
            else:
                proportion = round(numerator/denominator, 4)
            print(
                parent_node, "(p-value: " + str(node_p_value) + ")",
                "->", split_column, group,
                "(" + str(numerator) + "/" + str(denominator) + " = " + str(proportion*100) + "%)"
            )

