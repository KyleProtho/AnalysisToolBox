import pandas as pd
import numpy as np
import tensorflow as tf

def CreateCollaborativeFilteringRecommender(dataframe,
                                            outcome_variable,
                                            user_id_column,
                                            item_id_column,
                                            list_of_item_predictor_variables,):
    # Keep only the predictor variables
    dataframe = dataframe[[user_id_column, item_id_column, outcome_variable]].copy()
    
    # Pivot the dataframe on the user_id_column
    dataframe = dataframe.pivot(index=user_id_column, columns=item_id_column, values=outcome_variable)
    
    # # Return the model
    # return(model)
    return(None)
    
# Test the function
# Import data
# data_movies = pd.read_csv("C:/Users/oneno/Downloads/movies.csv")
data_ratings = pd.read_csv("C:/Users/oneno/Downloads/ratings.csv")
# Get average rating for each movie
df_temp = data_ratings[['movieId', 'rating']].groupby("movieId").mean()
df_temp = df_temp.rename(columns={'rating': 'avg_rating'})
data_ratings = data_ratings.merge(df_temp, on="movieId", how="left")
del(df_temp)
# Pivot the dataframe on the user_id_column
data_ratings = data_ratings.pivot(index='userId', columns='movieId', values='avg_rating')