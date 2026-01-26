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
    Generate and display a standardized summary table (Table 1) for baseline characteristics.

    This function creates a professional descriptive summary table, commonly 
    referred to as "Table 1" in clinical and academic research. it summarizes 
    continuous and categorical variables across different groups, providing 
    means, medians, frequencies, and optional p-values to assess baseline 
    comparability. The table can be rendered in multiple formats (HTML, 
    Markdown, LaTeX) for integration into different reporting environments.

    Summary tables (Table 1) are essential for:
      * Epidemiology: Summarizing demographic and clinical characteristics of a study cohort.
      * Healthcare: Comparing baseline health metrics between treatment and control patient groups.
      * Intelligence Analysis: Summarizing regional stability indicators across multiple geopolitical sectors.
      * Data Science: Inspecting the distribution and balance of features before model training.
      * Public Health: Monitoring the coverage of health services across different socioeconomic strata.
      * Social Science: Summarizing survey respondents' demographic profiles by education level.
      * Finance: Comparing asset performance metrics across different portfolio strategies.
      * Operations: Summarizing equipment performance metrics across several manufacturing sites.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame containing the variables to be summarized.
    value_column_name : str
        The name of the outcome or primary variable of interest.
    grouping_column_name : str
        The categorical column name used to group the rows (e.g., 'TreatmentGroup').
    list_of_row_variables : list of str
        The names of the columns to be included as rows in the summary table.
    table_format : {'html', 'grid', 'simple', 'github', 'latex'}, optional
        The output format for the table rendering. Defaults to 'html'.
    show_p_value : bool, optional
        Whether to calculate and display p-values for comparisons between groups. 
        Defaults to True.
    return_table_object : bool, optional
        Whether to return the underlying `tableone.TableOne` object for further 
        manipulation. Defaults to False.

    Returns
    -------
    tableone.TableOne or None
        Returns the TableOne object if `return_table_object` is True; 
        otherwise, returns None after displaying the table.

    Examples
    --------
    # Epidemiology: Summarizing cohort demographics by infection status
    import pandas as pd
    epi_df = pd.DataFrame({
        'Infected': ['Yes', 'No'] * 50,
        'Age': [45, 30, 60, 22] * 25,
        'Gender': ['M', 'F', 'F', 'M'] * 25,
        'BMI': [24.5, 28.1, 22.3, 31.5] * 25
    })
    RenderTableOne(
        epi_df, 'Age', 'Infected', 
        list_of_row_variables=['Age', 'Gender', 'BMI'],
        table_format='github'
    )

    # Intelligence Analysis: Regional stability summary
    intel_df = pd.DataFrame({
        'Sector': ['North', 'South'] * 30,
        'Stability Index': [0.8, 0.4] * 30,
        'Incident Rate': [5, 12] * 30,
        'Economic Risk': ['Low', 'High'] * 30
    })
    RenderTableOne(
        intel_df, 'Stability Index', 'Sector',
        list_of_row_variables=['Stability Index', 'Incident Rate', 'Economic Risk'],
        title_for_plot="Regional Geopolitical Stability Assessment"
    )

    # Healthcare: Baseline demographics for a clinical trial
    hosp_df = pd.DataFrame({
        'Group': ['Treatment', 'Control'] * 40,
        'Age': [55, 52] * 40,
        'Vitals_HR': [72, 75] * 40,
        'Diabetes': ['Yes', 'No'] * 40
    })
    RenderTableOne(
        hosp_df, 'Age', 'Group',
        list_of_row_variables=['Age', 'Vitals_HR', 'Diabetes'],
        table_format='html',
        show_p_value=True
    )
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

