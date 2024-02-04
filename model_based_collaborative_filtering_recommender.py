# -*- coding: utf-8 -*-
"""
Atharv Gupta
Recommend movies that are liked by similar users.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from read_and_analyze_movie_data_from_file import get_movielens_data


def collaborative_filtering_recommender(dataset, title): 
    # pivot table by movie titles with userId as columns and rating as values
    ratings = dataset.pivot_table(index='title', columns='userId', 
                                              values='rating').fillna(0)
    
    # substantiating an SVD object
    svd = TruncatedSVD(n_components = 200)
    matrix = svd.fit_transform(ratings)
    
    # converting to dataframe
    matrix_df = pd.DataFrame(matrix, index = dataset.title.unique().tolist())
    
    # finding a specific movie compared to others
    movies = np.array(matrix_df.loc[title]).reshape(1, -1)
    
    # score for the similarities
    score = cosine_similarity(matrix_df, movies).reshape(-1)
    
    # dataframe using the score for all the movies
    similar_df = pd.DataFrame(score, index = matrix_df.index, columns = ['collaborative'])
    
    # sorting by the score
    similar_df = similar_df.sort_values('collaborative', ascending=False)
    
    # returning the dataframe of similar movies
    return similar_df

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)    
    movielens = get_movielens_data()
    
    # TMDB doesn't have individual ratings of users so only doing
    #movielens recommendations
    movielens_recommendations = collaborative_filtering_recommender(movielens, 'Enigma (2001)')
    print("MovieLens top 10 recommendations")
    print(movielens_recommendations.head(10))
    print("")