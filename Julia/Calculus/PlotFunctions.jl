using Plots

function PlotFunctions(array_of_functions::Array, 
                       x_min::Number = -3,
                       x_max::Number = 3,
                       x_step::Number = 0.5)
    # Create x-axis
    x_values = x_min:x_step:x_max

    # Generate plot
    plot(x_values, array_of_functions)
end

# # Example
# f(x) = x^2
# g(x) = 2 * pi * x
# PlotFunctions([f])
# PlotFunctions([f, g])
PlotFunctions([f, g],
              x_min = -100, 
              x_max = 100)
