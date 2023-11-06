# Load packages
from explainerdashboard import ClassifierExplainer, RegressionExplainer, ExplainerDashboard

# Declare function
def CreateModelExplainerDashboard(model, 
                                  dataframe,
                                  outcome_variable,
                                  list_of_predictor_variables,
                                  # Dataset parameters
                                  test_size=0.2,
                                  random_seed=412,
                                  filter_nulls=True,
                                  predictor_descriptions=None,
                                  # Model parameters
                                  model_type='regression',
                                  numeric_outcome_units="",
                                  categorical_outcome_labels=None):
    # Ensure that predictor_descriptions is a dictionary
    if predictor_descriptions is not None and type(predictor_descriptions) != dict:
        raise TypeError("predictor_descriptions must be a dictionary, with the key being the predictor variable anme and the value being the description.")
    
    # Ensure that model type is either regression or classification
    if model_type not in ['regression', 'classification']:
        raise ValueError("model_type must be either 'regression' or 'classification'")
    
    # Ensure that categorical_outcome_labels is a list
    if categorical_outcome_labels is not None and type(categorical_outcome_labels) != list:
        raise TypeError("categorical_outcome_labels must be a list.")
    
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Drop rows with infinite values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing values if filter_nulls is True
    if filter_nulls:
        dataframe = dataframe.dropna()
    print("Count of examples eligible for inclusion in model training and testing:", len(dataframe.index))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create explainer dashboard
    if model_type == 'regression':
        explainer = RegressionExplainer(
            model=model, 
            X=test[list_of_predictor_variables], 
            y=test[outcome_variable],
            descriptions=predictor_descriptions, 
            units=numeric_outcome_units,
            target=outcome_variable
        )
    elif model_type == 'classification':
        explainer = ClassifierExplainer(
            model=model, 
            X=test[list_of_predictor_variables], 
            y=test[outcome_variable],
            descriptions=predictor_descriptions, 
            labels=categorical_outcome_labels,
            target=outcome_variable
        )
        
    # Show dashboard
    ExplainerDashboard(explainer).run()


# Test the function
from sklearn import datasets
exec(open("C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/Python/PredictiveAnalytics/CreateBoostedTreeModel.py").read())
iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
iris['species'] = datasets.load_iris(as_frame=True).target

# # NUMERICAL OUTCOME
# sep_len_boosted_tree_model = CreateBoostedTreeModel(
#     dataframe=iris,
#     outcome_variable='sepal length (cm)',
#     is_outcome_categorical=False,
#     list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#     # maximum_depth=5,
#     caption_for_model_test_performance_plot="The colored line shows the predicted values vs. the actual values in the test dataset. The grey straight line shows where the predicted values would be if the model was perfect."
# )
# CreateModelExplainerDashboard(
#     model=sep_len_boosted_tree_model,
#     dataframe=iris,
#     test_size=0.1,
#     model_type='regression',
#     outcome_variable='sepal length (cm)',
#     list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# )

# CATEGORICAL OUTCOME
species_boosted_tree_model = CreateBoostedTreeModel(
    dataframe=iris,
    outcome_variable='species',
    list_of_predictor_variables=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    # maximum_depth=5,
    caption_for_model_test_performance_plot="Each square shows the percentage of observations in the test dataset that are predicted to be in that category. The denominator the total number of actual outcomes in each category."
)
CreateModelExplainerDashboard(
    model=species_boosted_tree_model,
    dataframe=iris,
    model_type='classification',
    outcome_variable='species',
    list_of_predictor_variables=['sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    test_size=0.1
)
