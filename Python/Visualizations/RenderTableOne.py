import pandas as pd
from tableone import TableOne

def RenderTableOne(dataframe,
                   outcome_variable,
                   column_variable,
                   list_of_row_variables,
                   table_format='fancy_grid',
                   show_p_value=True,
                   return_table_object=False):
    # Select the columns to be included in the table
    dataframe = dataframe[[outcome_variable, column_variable] + list_of_row_variables]
    
    # Create table 1 object
    table_one = TableOne(dataframe, 
                         columns=list_of_row_variables,
                         groupby=column_variable, 
                         pval=show_p_value)
    
    # Render the table
    print(table_one.tabulate(tablefmt=table_format, 
                             floatfmt=".2f"))
    
    # Return the table object
    if return_table_object:
        return(table_one)
    
# Test function
from sklearn import datasets
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
iris['species'] = datasets.load_iris(as_frame=True).target
RenderTableOne(
    dataframe=iris,
    outcome_variable='sepal length (cm)',
    column_variable='species',
    list_of_row_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)
