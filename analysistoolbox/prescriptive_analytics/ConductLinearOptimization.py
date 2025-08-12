# Load packages
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linprog
import textwrap
import warnings

# Declare function
def ConductLinearOptimization(dataframe,
                              output_variable,
                              list_of_input_variables,
                              optimization_type="maximize",  # "maximize" or "minimize"
                              input_constraints=None,  # Dictionary with variable names as keys and (min, max) tuples as values
                              # Output arguments
                              print_optimization_summary=True,
                              print_constraint_summary=True,
                              # All plot arguments
                              data_source_for_plot=None,
                              # Optimization results plot arguments
                              plot_optimization_results=True,
                              dot_fill_color="#999999",
                              line_color="#b0170c",
                              figure_size_for_optimization_plot=(10, 6),
                              title_for_optimization_plot="Linear Optimization Results",
                              subtitle_for_optimization_plot="Optimal input values and resulting output value.",
                              caption_for_optimization_plot=None):
    """
    Conduct linear optimization to find optimal input values that maximize or minimize an output variable.
    
    This function uses linear programming to find the optimal combination of input variables
    that will maximize or minimize the output variable, subject to optional constraints on input values.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing input and output variables
    output_variable : str
        Name of the output variable to optimize
    list_of_input_variables : list
        List of input variable names to optimize
    optimization_type : str, optional
        Type of optimization: "maximize" or "minimize" (default: "maximize")
    input_constraints : dict, optional
        Dictionary with input variable names as keys and (min, max) tuples as values
        Example: {"variable1": (0, 10), "variable2": (None, 5)} where None means no constraint
    print_optimization_summary : bool, optional
        Whether to print optimization results summary (default: True)
    print_constraint_summary : bool, optional
        Whether to print constraint information (default: True)
    plot_optimization_results : bool, optional
        Whether to plot optimization results (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing optimization results including optimal values, objective value, and status
    """
    
    # Keep only the inputs and output variable
    dataframe = dataframe[list_of_input_variables + [output_variable]].copy()
    
    # Replace inf with nan, and drop rows with nan
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    
    if len(dataframe) < len(list_of_input_variables) + 1:
        raise ValueError(f"Insufficient data points ({len(dataframe)}) for {len(list_of_input_variables)} variables")
    
    # Fit linear regression to get coefficients for the objective function
    from sklearn.linear_model import LinearRegression
    
    X = dataframe[list_of_input_variables]
    y = dataframe[output_variable]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Prepare objective function coefficients
    if optimization_type.lower() == "maximize":
        # For maximization, we minimize the negative of the objective
        objective_coefficients = -coefficients
    else:
        # For minimization, we minimize the objective directly
        objective_coefficients = coefficients
    
    # Prepare bounds for input variables
    bounds = []
    for var in list_of_input_variables:
        if input_constraints and var in input_constraints:
            min_val, max_val = input_constraints[var]
            bounds.append((
                min_val if min_val is not None else -np.inf, 
                max_val if max_val is not None else np.inf
            ))
        else:
            # Use data range as bounds if no constraints specified
            var_min = dataframe[var].min()
            var_max = dataframe[var].max()
            bounds.append((var_min, var_max))
    
    # Solve the linear programming problem
    try:
        result = linprog(
            c=objective_coefficients,
            bounds=bounds,
            method='highs'  # Use the Highs solver for better performance
        )
        
        if result.success:
            optimal_inputs = result.x
            if optimization_type.lower() == "maximize":
                optimal_output = -result.fun + intercept
            else:
                optimal_output = result.fun + intercept
        else:
            warnings.warn(f"Optimization failed: {result.message}")
            optimal_inputs = None
            optimal_output = None
            
    except Exception as e:
        warnings.warn(f"Optimization error: {str(e)}")
        optimal_inputs = None
        optimal_output = None
    
    # Print optimization summary if requested
    if print_optimization_summary:
        print(f"\n{'='*60}")
        print(f"LINEAR OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Optimization Type: {optimization_type.upper()}")
        print(f"Output Variable: {output_variable}")
        print(f"Input Variables: {', '.join(list_of_input_variables)}")
        print(f"\nModel Coefficients:")
        for i, var in enumerate(list_of_input_variables):
            print(f"  {var}: {coefficients[i]:.6f}")
        print(f"Intercept: {intercept:.6f}")
        
        if optimal_inputs is not None:
            print(f"\nOptimal Solution:")
            for i, var in enumerate(list_of_input_variables):
                print(f"  {var}: {optimal_inputs[i]:.6f}")
            print(f"Optimal {output_variable}: {optimal_output:.6f}")
        else:
            print(f"\nNo optimal solution found.")
    
    # Print constraint summary if requested
    if print_constraint_summary and input_constraints:
        print(f"\n{'='*40}")
        print(f"CONSTRAINT SUMMARY")
        print(f"{'='*40}")
        for var, (min_val, max_val) in input_constraints.items():
            min_str = f"{min_val:.6f}" if min_val is not None else "-∞"
            max_str = f"{max_val:.6f}" if max_val is not None else "∞"
            print(f"{var}: [{min_str}, {max_str}]")
    
    # Plot optimization results if requested
    if plot_optimization_results and optimal_inputs is not None:
        # Set the size of the plot
        plt.figure(figsize=figure_size_for_optimization_plot)
        
        # Create subplot for input variables
        plt.subplot(1, 2, 1)
        
        # Create bar plot of optimal input values
        ax1 = sns.barplot(
            x=list_of_input_variables,
            y=optimal_inputs,
            color=dot_fill_color,
            alpha=0.7
        )
        
        # Add value labels on bars
        for i, v in enumerate(optimal_inputs):
            ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.title('Optimal Input Values', fontsize=12, color="#262626")
        plt.ylabel('Value', fontsize=10, color="#666666")
        plt.xticks(rotation=45, ha='right')
        
        # Remove spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color('#666666')
        ax1.spines['left'].set_color('#666666')
        
        # Create subplot for output variable
        plt.subplot(1, 2, 2)
        
        # Create bar plot of optimal output value
        ax2 = sns.barplot(
            x=[output_variable],
            y=[optimal_output],
            color=line_color,
            alpha=0.7
        )
        
        # Add value label on bar
        ax2.text(0, optimal_output, f'{optimal_output:.3f}', ha='center', va='bottom')
        
        plt.title(f'Optimal {output_variable}', fontsize=12, color="#262626")
        plt.ylabel('Value', fontsize=10, color="#666666")
        
        # Remove spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color('#666666')
        ax2.spines['left'].set_color('#666666')
        
        # Add main title
        plt.suptitle(
            title_for_optimization_plot,
            fontsize=14,
            color="#262626",
            y=0.95
        )
        
        # Add subtitle
        plt.figtext(
            0.5, 0.90,
            subtitle_for_optimization_plot,
            fontsize=11,
            color="#666666",
            ha='center'
        )
        
        # Add caption if provided
        if caption_for_optimization_plot or data_source_for_plot:
            caption_text = ""
            if caption_for_optimization_plot:
                caption_text = caption_for_optimization_plot
            if data_source_for_plot:
                caption_text += f"\n\nSource: {data_source_for_plot}"
            
            plt.figtext(
                0.5, 0.02,
                caption_text,
                fontsize=8,
                color="#666666",
                ha='center'
            )
        
        plt.tight_layout()
        plt.show()
    
    
    
    # Return results
    results = {
        'success': optimal_inputs is not None,
        'optimization_type': optimization_type,
        'output_variable': output_variable,
        'input_variables': list_of_input_variables,
        'coefficients': coefficients,
        'intercept': intercept,
        'optimal_inputs': optimal_inputs,
        'optimal_output': optimal_output,
        'constraints': input_constraints,
        'model': model
    }
    
    return results


# # Test the function
# # Sample data
# data = pd.DataFrame({
#     'input1': [1, 2, 3, 4, 5],
#     'input2': [2, 4, 6, 8, 10],
#     'output': [10, 20, 30, 40, 50]
# })

# # Define constraints (optional)
# constraints = {
#     'input1': (0, 10),  # input1 between 0 and 10
#     'input2': (None, 15)  # input2 maximum 15, no minimum
# }

# # Run optimization
# results = ConductLinearOptimization(
#     dataframe=data,
#     output_variable='output',
#     list_of_input_variables=['input1', 'input2'],
#     optimization_type='maximize',
#     input_constraints=constraints
# )
