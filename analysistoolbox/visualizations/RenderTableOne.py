# Load packages
from IPython.display import display, Markdown, HTML, Latex
import pandas as pd

# Declare function
def RenderTableOne(dataframe,
                   value_column_name,
                   grouping_column_name,
                   list_of_row_variables,
                   table_format='html',
                   show_p_value=True,
                   return_table_object=False):
    """
    Generates and displays a summary table (Table 1) from a given pandas DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame that contains the data.
        value_column_name (str): The name of the column in the DataFrame to be used as the outcome variable.
        grouping_column_name (str): The name of the column in the DataFrame to be used as the grouping variable.
        list_of_row_variables (list): A list of column names to be included as rows in the table.
        table_format (str, optional): The format of the table. Defaults to 'html'.
        show_p_value (bool, optional): Whether to include p-values in the table. Defaults to True.
        return_table_object (bool, optional): Whether to return the TableOne object. Defaults to False.

    Returns:
        tableone.TableOne: The TableOne object, if return_table_object is True. Otherwise, None.
    """
    # Lazy load uncommon packages
    from tableone import TableOne
    
    # Select the columns to be included in the table
    dataframe = dataframe[[value_column_name, grouping_column_name] + list_of_row_variables]
    
    # Create table 1 object
    table_one = TableOne(dataframe, 
                         columns=list_of_row_variables,
                         groupby=grouping_column_name, 
                         pval=show_p_value)
    
    # Render the table
    if table_format in ['html']:
        table_one_display = table_one.tabulate(
            tablefmt=table_format, 
            floatfmt=".2f"
        )
        table_one_display = HTML(table_one_display)
        display(table_one_display)
    elif table_format in ['grid', 'simple', 'github']:
        table_one_display = table_one.tabulate(
            tablefmt=table_format, 
            floatfmt=".2f"
        )
        table_one_display = Markdown(table_one_display)
        display(table_one_display)
    elif table_format in ['latex_booktabs', 'latex']:
        table_one_display = table_one.tabulate(
            tablefmt=table_format, 
            floatfmt=".2f"
        )
        table_one_display = Latex(table_one_display)
        display(table_one_display)
    else:
        table_one_display = table_one.tabulate(
            tablefmt=table_format, 
            floatfmt=".2f"
        )
        print(table_one_display)
    
    # Return the table object
    if return_table_object:
        return(table_one)

