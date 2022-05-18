using Plots, CalculusWithJulia, Calculus

function FindDerivative(f_of_x::Function, x_limit::Number = 0; 
                        from_left::Bool = true,
                        x_min::Number = -3,
                        x_max::Number = 3,
                        x_step::Number = 0.5)
    # Create x-axis
    x_values = x_min:x_step:x_max

    # Generate plot
    p = plot(x_values, f_of_x,
             label = f_of_x);

    # Iterate through bases and approach limit
    arr_bases = [1, 2, 10, 100, 1000]
    arr_colors = ["#ffe5ee", "#ffbdd5", "#ff92b9", "#ff659c", "#ed116a"]
    for i in 1:length(arr_bases)
        if from_left
            current_point = x_limit - (x_step / arr_bases[i])
        else
            current_point = x_limit + (x_step / arr_bases[i])
        end
        plot!(tangent(f, current_point), 
              color = arr_colors[i],
              label = round(current_point, digits = 3));
    end

    # Display plot
    display(p)

    # Return derivative
    return derivative(f, x_limit)
end

# # Example
# f(x) = x^2
# FindDerivative(f, 0)
# FindDerivative(f, 0, from_left = false)
# FindDerivative(f, -1.5)
# FindDerivative(f, -1.5, from_left = false)
# FindDerivative(f, 1.5)
# FindDerivative(f, 1.5, from_left = false)