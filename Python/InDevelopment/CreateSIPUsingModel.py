from random import random
import pandas as pd
import statsmodels.api as sm
from tensorflow import keras

def CreateSIPUsingModel(model_object_or_dictionary,
                        sip_dataframe,
                        outcome_variable,
                        list_of_predictor_variables):
    # If model_object is a dictionary, then scale the data and make predictions. Otherwise, just make predictions.
    if type(model_object) is dict:
        # Check scaler is a normalization layer
        if str(type(model_object['scaler'])) == "<class 'keras.layers.preprocessing.normalization.Normalization'>":
            normalizer = model_object['scaler']
            norm_predictor_variables = normalizer(np.array(sip_dataframe[list_of_predictor_variables]))
            sip_dataframe[list_of_predictor_variables] = pd.DataFrame(norm_predictor_variables.numpy(), columns=list_of_predictor_variables)
            sip_dataframe[outcome_variable] = model_object['model'].predict(sip_dataframe[list_of_predictor_variables])
        else:
            sip_dataframe[outcome_variable] = model_object['model'].predict(
                model_object['scaler'].transform(sip_dataframe[list_of_predictor_variables])
            )  
    else:
        sip_dataframe[outcome_variable] = model_object.predict(sip_dataframe[list_of_predictor_variables])
        
    # Return the dataframe with predictions
    return(sip_dataframe)

# # Test function
# # Import functions
# exec(open('C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/Python/Simulations/CreateSIPDataframe.py').read())
# exec(open('C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/Python/Simulations/SimulateNormalDistributionFromSample.py').read())
# exec(open('C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/Python/Predictive Analytics/CreateLinearRegressionModel.py').read())
# exec(open('C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/Python/Predictive Analytics/CreateBoostedTreeModel.py').read())
# exec(open('C:/Users/oneno/OneDrive/Creations/Snippets for Statistics/SnippetsForStatistics/Python/Predictive Analytics/CreateNeuralNetwork_SingleOutcome.py').read())
# # Import dataset
# from sklearn import datasets
# iris = pd.DataFrame(datasets.load_iris(as_frame=True).data)
# # Create list of predictors
# list_of_predictors = ['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# # Create SIP dataframe
# sip_iris = CreateSIPDataframe(name_of_items='iris_id',
#                               list_of_items=range(1),
#                               number_of_trials=10000)
# # Create normal distribution for each predictor
# for predictor in list_of_predictors:
#     df_temp = SimulateNormalDistributionFromSample(dataframe=iris,
#                                                    outcome_variable=predictor,
#                                                    trials=10000,
#                                                    plot_sim_results=False)
#     sip_iris[predictor] = df_temp[predictor]
# del(df_temp, predictor)
# # Test regression model
# # Create regression model
# sepal_length_reg_model = CreateLinearRegressionModel(
#     dataframe=iris,
#     outcome_variable='sepal length (cm)',
#     list_of_predictor_variables=list_of_predictors,
#     scale_predictor_variables=True
# )
# # Create estimates using regression model
# sip_iris = CreateSIPUsingModel(
#     model_object=sepal_length_reg_model,
#     sip_dataframe=sip_iris,
#     outcome_variable='sepal length (cm) - Regression Model',
#     list_of_predictor_variables=list_of_predictors
# )
# # Test boosted tree model
# # Create boosted tree model
# sepal_length_boosted_tree_model = CreateBoostedTreeModel(
#     dataframe=iris,
#     outcome_variable='sepal length (cm)',
#     list_of_predictor_variables=list_of_predictors,
#     is_outcome_categorical=False
# )
# # Create estimates using boosted tree model
# sip_iris = CreateSIPUsingModel(
#     model_object=sepal_length_boosted_tree_model,
#     sip_dataframe=sip_iris,
#     outcome_variable='sepal length (cm) - Boosted Tree Model',
#     list_of_predictor_variables=list_of_predictors
# )
# # Test neural network model
# sepal_length_nn_model = CreateNeuralNetwork_SingleOutcome(
#     dataframe=iris,
#     outcome_variable='sepal length (cm)',
#     list_of_predictor_variables=list_of_predictors,
#     number_of_hidden_layers=2,
#     is_outcome_categorical=False
# )
# sip_iris = CreateSIPUsingModel(
#     model_object=sepal_length_nn_model,
#     sip_dataframe=sip_iris,
#     outcome_variable='sepal length (cm) - Neural Network Model',
#     list_of_predictor_variables=list_of_predictors
# )
