using Plots

function PlotSystem(array_of_equations::Array, 
                    x_min::Number = -10, 
                    x_max::Number = 10, 
                    x_step::Number = 0.5)
    # Create x-axis 
    x_values = x_min:x_step:x_max

    # Generate plot
    plot(x_values, array_of_equations)
end

# # Example
# f(x) = x + 0
# g(x) = -2x + 3
# PlotSystem([f, g])
