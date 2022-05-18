
function SolveSystemOfEquations(matrix_of_coefficients::Matrix, 
                                array_of_intercepts::Vector)
    # Create solution array
    solution = matrix_of_coefficients\array_of_intercepts

    # Return solution 
    return solution
end

# # Example
# # -x + y = 0
# # 2x + y = 3
# SolveSystemOfEquations(
#     [-1 1; 2 1],
#     [0, 3]
# )
