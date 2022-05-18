using Plots

function PlotFunctions(array_of_functions::Array; 
                       x_min::Number = -3,
                       x_max::Number = 3,
                       x_step::Number = 0.5,
                       x_limit_vline::Union{Number, Nothing} = nothing)
    # Create x-axis
    x_values = x_min:x_step:x_max

    # Generate plot
    p = plot(x_values, array_of_functions)

    # Draw limit if specified
    if x_limit_vline != nothing
        vline!([x_limit_vline], 
               seriestype="vline",
               linstyle="dash",
               color="red")
    end

    # Show plot
    display(p)
end

# # Example
# f(x) = x^3
# g(x) = 3 * x^2
# h(x) = 1 / (x - 3)
# PlotFunctions([f])
# PlotFunctions([f, g])
# PlotFunctions([f, g],
#               x_min = -100, 
#               x_max = 100)
# PlotFunctions([f],
#               x_min = -15, 
#               x_max = 15,
#               x_limit_vline = 3)
# PlotFunctions([h],
#               x_min = -50, 
#               x_max = 50,
#               x_limit_vline = 3)
