import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

def CreateContentBasedRecommender(dataframe,
                                  outcome_variable,
                                  user_list_of_predictor_variables,
                                  item_list_of_predictor_variables,
                                  user_number_of_hidden_layers=2,
                                  item_number_of_hidden_layers=2,
                                  number_of_recommendations=10,
                                  test_size=0.2,
                                  scale_variables=True,
                                  plot_loss=True,
                                  plot_model_test_performance=True,
                                  show_predictor_ranges=False,
                                  initial_learning_rate=0.01,
                                  number_of_steps_gradient_descent=50,
                                  lambda_for_regularization=0.001,
                                  random_seed=412):
    
    # Keep only the predictors from the user and item dataframes
    dataframe = dataframe[user_list_of_predictor_variables + item_list_of_predictor_variables + [outcome_variable]].copy()
    
    # Set the random seed
    tf.random.set_seed(random_seed)
    
    # Use Keras to create a normalization layer
    if scale_variables:
        # For users
        scalerUser = StandardScaler()
        scalerUser.fit(dataframe[user_list_of_predictor_variables])
        dataframe[user_list_of_predictor_variables] = scalerUser.transform(dataframe[user_list_of_predictor_variables])
        # For items
        scalerItem = StandardScaler()
        scalerItem.fit(dataframe[item_list_of_predictor_variables])
        dataframe[item_list_of_predictor_variables] = scalerItem.transform(dataframe[item_list_of_predictor_variables])
        # Outcome variable
        scalerOutcome = MinMaxScaler((-1, 1))
        scalerOutcome.fit(dataframe[outcome_variable].values.reshape(-1,1))
        dataframe[outcome_variable] = scalerOutcome.transform(dataframe[outcome_variable].values.reshape(-1,1))
        
    # Show the peak-to-peak range of each predictor
    if show_predictor_ranges:
        print("\nPeak-to-peak range of each predictor in user dataset:")
        print(np.ptp(dataframe[user_list_of_predictor_variables], axis=0))
        print("\nPeak-to-peak range of each predictor in item dataset:")
        print(np.ptp(dataframe[item_list_of_predictor_variables], axis=0))
    
    # Split dataframe into training and test sets
    train, test = train_test_split(
        dataframe, 
        test_size=test_size,
        shuffle=True,
        random_state=random_seed
    )
    
    # Create dictionary of layers to be used in the neural network
    user_dict_layers = {}
    item_dict_layers = {}
    
    # Create hidden layers
    # # For users
    for i in range(user_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        if i < 10:
            layer_name = 'layer_0' + str(i + 1)
        else:
            layer_name = 'layer_' + str(i + 1)
        user_dict_layers[key_text] = tf.keras.layers.Dense(
            number_of_recommendations + ((user_number_of_hidden_layers - 1 - i) * 10),
            activation='relu', 
            name=layer_name,
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
    # # For items
    for i in range(item_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        if i < 10:
            layer_name = 'layer_0' + str(i + 1)
        else:
            layer_name = 'layer_' + str(i + 1)
        item_dict_layers[key_text] = tf.keras.layers.Dense(
            number_of_recommendations + ((item_number_of_hidden_layers - 1 - i) * 10),
            activation='relu', 
            name=layer_name,
            kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
        )
        
    # Create output layer
    # # For users
    user_dict_layers['Output layer'] = tf.keras.layers.Dense(
        number_of_recommendations, 
        activation='linear', 
        name='final_layer',
        kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
    )
    # # For items
    item_dict_layers['Output layer'] = tf.keras.layers.Dense(
        number_of_recommendations, 
        activation='linear', 
        name='final_layer',
        kernel_regularizer=tf.keras.regularizers.l2(lambda_for_regularization)
    )
    
    # Create list of layers to be used in the neural networks
    # # For users
    user_list_of_layers = []
    for i in range(user_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        user_list_of_layers.append(user_dict_layers[key_text])
    user_list_of_layers.append(user_dict_layers['Output layer'])
    # # For items
    item_list_of_layers = []
    for i in range(item_number_of_hidden_layers):
        key_text = 'Hidden layer ' + str(i + 1)
        item_list_of_layers.append(item_dict_layers[key_text])
    item_list_of_layers.append(item_dict_layers['Output layer'])
    
    # Create sequential models
    user_model = tf.keras.Sequential(user_list_of_layers)
    item_model = tf.keras.Sequential(item_list_of_layers)
    
    # Create input layers and point to base layers
    # # For users
    user_input = tf.keras.layers.Input(shape=(len(user_list_of_predictor_variables),))
    user_output = user_model(user_input)
    user_output = tf.linalg.l2_normalize(user_output, axis=1)
    # # For items
    item_input = tf.keras.layers.Input(shape=(len(item_list_of_predictor_variables),))
    item_output = item_model(item_input)
    item_output = tf.linalg.l2_normalize(item_output, axis=1)
    
    # Create recommendation output layer
    recommender_output = tf.keras.layers.Dot(axes=1)([user_output, item_output])
    
    # Create model
    model = tf.keras.Model([user_input, item_input], recommender_output)
    
    # Show model summary
    model.summary()
    
    # Define loss function and optimizer
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        metrics=['mean_squared_error']
    )
    
    # Train the model
    loss_history = model.fit(
        [train[user_list_of_predictor_variables].values, train[item_list_of_predictor_variables].values],
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
        
    # # Evaluate the model
    # model.evaluate(
    #     [test[user_list_of_predictor_variables].values, test[item_list_of_predictor_variables].values],
    #     test[outcome_variable].values
    # )
    
    # Make predictions on test set
    predictions = model.predict([test[user_list_of_predictor_variables].values, test[item_list_of_predictor_variables].values])
    if scale_variables:
        predictions = scalerOutcome.inverse_transform(predictions.reshape(-1, 1))
        test[outcome_variable] = scalerOutcome.inverse_transform(test[outcome_variable].values.reshape(-1, 1))
    test['Predicted'] = predictions
    
    # Plot prediction performance if requested
    if plot_model_test_performance:
        plt.figure(figsize=(9,9))
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
            sns.regplot(
                data=test,
                x=outcome_variable,
                y='Predicted'
            )
            plt.plot(test[outcome_variable], test[outcome_variable], color='black', alpha=0.35)
            plt.title('Predicted vs. Observed Outcome', size = 15)
        plt.show()
        
    # Create dictionary of objects to be returned
    if scale_variables:
        dict_return = {
            'User Scaler': scalerUser,
            'Item Scaler': scalerItem,
            'Outcome Scaler': scalerOutcome,
            'Model': model
        }
        return(dict_return)
    else:
        return(model)

# # Test the function
# # Import data
# data_movies = pd.read_csv("C:/Users/oneno/Downloads/movies.csv")
# data_users = pd.read_csv("C:/Users/oneno/Downloads/ratings.csv")
# # Add genre flags
# data_movies['Movie genre - Animated'] = data_movies['genres'].str.contains('Animation').astype(int)
# data_movies['Movie genre - Comedy'] = data_movies['genres'].str.contains('Comedy').astype(int)
# data_movies['Movie genre - Drama'] = data_movies['genres'].str.contains('Drama').astype(int)
# data_movies['Movie genre - Romance'] = data_movies['genres'].str.contains('Romance').astype(int)
# data_movies['Movie genre - Thriller'] = data_movies['genres'].str.contains('Thriller').astype(int)
# data_movies['Movie genre - Action'] = data_movies['genres'].str.contains('Action').astype(int)
# data_movies['Movie genre - Adventure'] = data_movies['genres'].str.contains('Adventure').astype(int)
# data_movies['Movie genre - Fantasy'] = data_movies['genres'].str.contains('Fantasy').astype(int)
# data_movies['Movie genre - Sci-fi'] = data_movies['genres'].str.contains('Sci-Fi').astype(int)
# data_movies['Movie genre - Horror'] = data_movies['genres'].str.contains('Horror').astype(int)
# # Join average rating to movies
# df_temp = data_users.groupby('movieId').agg({'rating': 'mean'}).reset_index()
# df_temp.rename(columns={'rating': 'Movie - Average rating'}, inplace=True)
# data_movies = data_movies.merge(df_temp, how='left', on='movieId')
# del(df_temp)
# # Create list of genre columns
# list_of_genres = [
#     'Movie genre - Animated', 'Movie genre - Comedy', 'Movie genre - Drama', 
#     'Movie genre - Romance', 'Movie genre - Thriller', 'Movie genre - Action', 
#     'Movie genre - Adventure', 'Movie genre - Fantasy', 'Movie genre - Sci-fi', 
#     'Movie genre - Horror'
# ]
# # Join movie attributes to users
# data_users = data_users.merge(
#     data_movies[['movieId', 'Movie - Average rating'] + list_of_genres],
#     how='left', 
#     on='movieId'
# )
# del(data_movies)
# # Left join average rating by genre
# list_user_genre_ratings = []
# for genre in list_of_genres:
#     new_col_name = genre.replace('Movie genre - ', 'User - Average rating for ')
#     list_user_genre_ratings.append(new_col_name)
#     data_users[new_col_name] = data_users[genre] * data_users['rating']
# data_users_summary = data_users.groupby('userId').agg({col: 'mean' for col in list_user_genre_ratings}).reset_index()
# data_users.drop(columns=list_user_genre_ratings, inplace=True)
# data_users = data_users.merge(data_users_summary, how='left', on='userId')
# del(data_users_summary, genre, new_col_name, list_user_genre_ratings, list_of_genres)
# # Create content based recommender
# movie_recommnder = CreateContentBasedRecommender(
#     dataframe=data_users,
#     outcome_variable='rating',
#     user_list_of_predictor_variables=[
#         'User - Average rating for Animated',
#         'User - Average rating for Comedy', 'User - Average rating for Drama',
#         'User - Average rating for Romance',
#         'User - Average rating for Thriller',
#         'User - Average rating for Action',
#         'User - Average rating for Adventure',
#         'User - Average rating for Fantasy', 'User - Average rating for Sci-fi',
#         'User - Average rating for Horror'
#     ],
#     item_list_of_predictor_variables=[
#         'Movie - Average rating',
#         'Movie genre - Animated', 'Movie genre - Comedy', 'Movie genre - Drama',
#         'Movie genre - Romance', 'Movie genre - Thriller',
#         'Movie genre - Action', 'Movie genre - Adventure',
#         'Movie genre - Fantasy', 'Movie genre - Sci-fi', 'Movie genre - Horror'
#     ]
# )
