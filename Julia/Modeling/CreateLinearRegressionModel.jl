# Load packages
using Random, Statistics, GLM, DataFrames, DataFramesMeta

# Define CreateLinearRegressionModel function
function CreateLinearRegressionModel(
    dataframe::DataFrame, 
    outcome_var::String, 
    arr_predictor_vars::Vector{String},
    test_set_size::Float64=.20,
    show_help::Bool=false
)
    # Check for valid arguments
    @assert 0 <= test_set_size <= 1
    
    # Subselect variables
    arr_variables = copy(arr_predictor_vars)
    push!(arr_variables, outcome_var)
    df_regression = @chain dataframe begin
        @select $arr_variables
    end 

    # Filter to complete cases 
    dropmissing!(df_regression)

    # Split into test and training dataset
    ids = collect(axes(df_regression, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df_regression) .* test_set_size
    df_regression_train = view(df_regression, sel, :)
    df_regression_test = view(df_regression, .!sel, :)

    # Create formula string
    predictors = term.((arr_predictor_vars))
    formula_string = term(outcome_var) ~ foldl(+, predictors)

    # Create linear model
    regression_model = lm(formula_string, df_regression_train)

    # If requested, show diagnostic plots

    # If requestion, show help text 
    if show_help
        println(
            "\nThe ouput of the CreateLinearRegressionModel function is a dictionary containing the regression results, a test dataset of predictors, and a test dataset of outcomes.",
            "\n\t--To access the linear regression model, use the 'Fitted Model' key.",
            "\n\t--To access the test dataset of predictors, use the 'Predictor Test Dataset' key.",
            "\n\t--To access the test dataset of outcomes, use the 'Outcome Test Dataset' key.",
            #"\n\nRemember that you can utilize the TestLinearRegressionModelCI function to test the accuracy of the 95% confidence interval of your fitted regression model."
        )
    end

    # Return model and datasets
    return_dict = Dict(
        "Fitted Model" => regression_model, 
        "Training Dataset" => df_regression_train,
        "Test Dataset" => df_regression_test
    )
    return(return_dict)
end



# # For development: Import iris dataset using RDatasets
# using RDatasets
# iris_data = dataset("datasets", "iris")
# lm_dictionary = CreateLinearRegressionModel(
#     iris_data,
#     "SepalLength",
#     [
#         "PetalLength", 
#         "PetalWidth"  # Eliminated predictor #1
#     ]
# )
