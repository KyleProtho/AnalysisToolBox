# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Declare function
def CreateNeuralNetwork_SingleOutcome(dataframe,
                                      outcome_variable,
                                      list_of_predictor_variables,
                                      number_of_hidden_layers,
                                      is_outcome_categorical=True,
                                      # Model training arguments
                                      test_size=0.2,
                                      scale_predictor_variables=True,
                                      initial_learning_rate=0.01,
                                      number_of_steps_gradient_descent=100,
                                      lambda_for_regularization=0.001,
                                      random_seed=412,
                                      # Output arguments
                                      print_peak_to_peak_range_of_each_predictor=False,
                                      plot_loss=True,
                                      plot_model_test_performance=True):
    """
    Construct, train, and evaluate a deep neural network for binary/multi-class classification or regression.

    This function leverages TensorFlow and Keras to build a fully connected 
    Sequential neural network tailored for tabular data. It automates the 
    creation of dynamic hidden layer architectures, handles feature 
    normalization via Keras layers, and supports diverse objective functions 
    including Binary Crossentropy, Sparse Categorical Crossentropy, and 
    Mean Squared Error.

    Neural networks are essential for:
      * Detecting complex, non-linear patterns in high-dimensional financial or security data
      * Predicting credit risk or fraud where interactions between features are non-linear
      * Classifying multi-category intelligence reports or customer segments
      * Forecasting continuous variables like market volatility or housing prices
      * Building multi-class classifiers for sentiment analysis or product categorization
      * Modeling operational throughput in complex manufacturing or supply chain environments
      * Identifying subtle behavioral trends in large-scale consumer or medical datasets

    The function provides a simplified interface for standard deep learning 
    tasks, including automated loss curve plotting and performance 
    diagnostics (confusion matrices or regression plots). It also implements 
    L2 regularization to prevent overfitting in deep architectures.

    Parameters
    ----------
    dataframe
        The input pandas.DataFrame containing the feature set and target variable.
    outcome_variable
        The name of the target column to be predicted.
    list_of_predictor_variables
        A list of column names to be used as input features (X).
    number_of_hidden_layers
        The number of dense layers to insert between the input and output. 
        Higher values increase the capacity to model complex relationships.
    is_outcome_categorical
        If True, treats the task as classification. If False, treats it as 
        regression. Defaults to True.
    test_size
        The proportion of data reserved for model evaluation. Defaults to 0.2.
    scale_predictor_variables
        If True, utilizes a Keras `Normalization` layer to standardize feature 
        inputs. Recommended for neural network stability. Defaults to True.
    initial_learning_rate
        The step size for the Adam optimizer. Small values are more stable 
        but slower to converge. Defaults to 0.01.
    number_of_steps_gradient_descent
        The number of training epochs. Defaults to 100.
    lambda_for_regularization
        The L2 penalty weight applied to all dense layers to mitigate 
        overfitting. Defaults to 0.001.
    random_seed
        The integer seed used for TensorFlow's random state and data 
        partitioning. Defaults to 412.
    print_peak_to_peak_range_of_each_predictor
        If True, prints the scale of the predictor variables to help assess 
        data range. Defaults to False.
    plot_loss
        Whether to render the training loss curve as a function of epochs. 
        Defaults to True.
    plot_model_test_performance
        Whether to generate a diagnostic plot (Confusion Matrix, Probability 
        Scatter, or Regplot) for the test dataset. Defaults to True.

    Returns
    -------
    tf.keras.Model or dict
        If `scale_predictor_variables` is False, returns the fitted Keras 
        Sequential model. If True, returns a dictionary containing the 'model' 
        and the 'scaler' (normalization layer).

    Examples
    --------
    # Create a 3-layer deep classifier to predict customer attrition
    model = CreateNeuralNetwork_SingleOutcome(
        df, 
        outcome_variable='attrition', 
        list_of_predictor_variables=['salary', 'tenure', 'hours'],
        number_of_hidden_layers=3
    )

    # Build a regression network for home price estimation with 5 hidden layers
    results = CreateNeuralNetwork_SingleOutcome(
        real_estate_df,
        outcome_variable='price',
        list_of_predictor_variables=['sqft', 'lot_size', 'age'],
        number_of_hidden_layers=5,
        is_outcome_categorical=False,
        number_of_steps_gradient_descent=500
    )

    """
    
    # Lazy load uncommon packages
    import tensorflow as tf
    
    # Keep only the predictors and outcome variable
    dataframe = dataframe[list_of_predictor_variables + [outcome_variable]].copy()
    
    # Set the random seed
    tf.random.set_seed(random_seed)
    
    # Use Keras to create a normalization layer
    if scale_predictor_variables:
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(dataframe[list_of_predictor_variables]))  # Learns the statistics of the data
        norm_predictor_variables = normalizer(np.array(dataframe[list_of_predictor_variables]))
        dataframe[list_of_predictor_variables] = pd.DataFrame(norm_predictor_variables.numpy(), columns=list_of_predictor_variables)

    # Show the peak-to-peak range of each predictor
    if print_peak_to_peak_range_of_each_predictor:
        print("\nPeak-to-peak range of each predictor:")
        print(np.ptp(dataframe[list_of_predictor_variables], axis=0))
        
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        random_state=random_seed
    )
    
    # Choose activation function
    if is_outcome_categorical: 
        if len(dataframe[outcome_variable].unique()) == 2:
            activation_function = 'sigmoid'
        else:
            activation_function = 'linear'
    else:
        if dataframe[outcome_variable].min() >= 0:
            activation_function = 'relu'
        else:
            activation_function = 'linear'
                
    # Create dictionary of layers to be used in the neural network
    dict_layers = {}
    
    # Create input layer
    dict_layers['Input layer'] = tf.keras.layers.Input(
        shape=(len(list_of_predictor_variables),), 
        name='input_layer'
    )
    
    # Create hidden layers
    for i in range(number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        if i < 10:
            layer_name = 'layer_0' + str(i + 1)
        else:
            layer_name = 'layer_' + str(i + 1)
        dict_layers[key_text] = tf.keras.layers.Dense(
            10 + ((number_of_hidden_layers - 1 - i) * 10), 
            activation='relu', 
            name=layer_name,
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
        
    # Create output layer
    if is_outcome_categorical and len(dataframe[outcome_variable].unique()) > 2:
        dict_layers['Output layer'] = tf.keras.layers.Dense(
            len(dataframe[outcome_variable].unique()), 
            activation=activation_function, 
            name='softmax_layer',
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
    else:
        dict_layers['Output layer'] = tf.keras.layers.Dense(
            1, 
            activation=activation_function, 
            name='final_layer',
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
    
    # Create list of layers to be used in the neural network
    list_of_layers = [dict_layers['Input layer']]
    for i in range(number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        list_of_layers.append(dict_layers[key_text])
    list_of_layers.append(dict_layers['Output layer'])
    
    # Create sequential model
    model = tf.keras.Sequential(list_of_layers) 
    
    # Show model summary
    model.summary()
    
    # Define loss function and optimizer
    if is_outcome_categorical:
        if len(dataframe[outcome_variable].unique()) == 2:
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                metrics=['accuracy']
            )
        else:
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                metrics=['accuracy']
            )
    else:
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            metrics=['mean_squared_error']
        )
    
    # Train the model
    loss_history = model.fit(
        train[list_of_predictor_variables].values,
        train[outcome_variable].values,
        epochs=number_of_steps_gradient_descent,
        verbose=0
    )
    if plot_loss:
        plt.figure(figsize=(9,9))
        sns.lineplot(
            x=range(1, number_of_steps_gradient_descent + 1),
            y=loss_history.history['loss']
        )
        plt.title('Loss Curve', size = 15)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    # Test the model
    predictions = model.predict(test[list_of_predictor_variables].values)
    if is_outcome_categorical and len(dataframe[outcome_variable].unique()) > 2:
        predictions = tf.nn.softmax(predictions).numpy()  # Convert to probabilities
        predictions = pd.DataFrame(predictions).idxmax(axis=1).values  # Convert to class labels
    test['Predicted'] = predictions
    
    # Plot results of model test
    if plot_model_test_performance:
        plt.figure(figsize=(9,9))
        if is_outcome_categorical:
            if len(dataframe[outcome_variable].unique()) == 2:
                sns.scatterplot(
                    data=test,
                    x=outcome_variable,
                    y='Predicted'
                )
                plt.title('Probability vs. Observed Outcome', size = 15)
                plt.ylabel('Predicted probability')
                plt.ylim(0, 1.05)
            else:
                confusion_matrix = metrics.confusion_matrix(
                    test[outcome_variable], 
                    test['Predicted']
                )
                sns.heatmap(
                    confusion_matrix, 
                    annot=True, 
                    fmt=".3f", 
                    linewidths=.5, 
                    square=True, 
                    cmap='Blues_r'
                )
                plt.title('Confusion Matrix', size = 15)
                plt.ylabel('Actual label')
                plt.xlabel('Predicted label')
        else:
            sns.regplot(
                data=test,
                x=outcome_variable,
                y='Predicted'
            )
            plt.plot(test[outcome_variable], test[outcome_variable], color='black', alpha=0.35)
            plt.title('Predicted vs. Observed Outcome', size = 15)
        plt.show()
    
    # Return the model
    if scale_predictor_variables:
        dict_return = {
            'scaler': normalizer,
            'model': model
        }
        return(dict_return)
    else:
        return(model)

