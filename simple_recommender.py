# -*- coding: utf-8 -*-
"""
Atharv Gupta
Offers generalized recommendations to all the users, it suggests movies
that are popular and have a high probability of being liked by everyone
"""
import pandas as pd
from read_and_analyze_movie_data_from_file import get_movielens_data
from read_and_analyze_movie_data_from_file import get_tmdb_data


def simple_recommender(dataset, dataset_name):    
    # fiding the mean of our rating by title
    ratings_mean = pd.DataFrame(dataset.groupby('title')['rating'].mean())
    
    # counting how many users rated the title
    ratings_mean['rating_counts'] = pd.DataFrame(dataset.groupby('title')['rating'].count())

    # if tmdb dataset is used then we use the existing rating_counts column
    if dataset_name == 'tmdb':
        ratings_mean['rating_counts'] = pd.DataFrame(dataset.groupby('title')['rating_counts'].sum())

    C = ratings_mean['rating'].mean()
    
    # We will use cutoff m as the 90th percentile, so for a movie to be recommended
    # it must have more votes than at least 90% of the movies on the list
    m = ratings_mean['rating_counts'].quantile(0.9)
    
    # df with the required minimum vote
    weighted_ratings_mean = ratings_mean.copy().loc[ratings_mean['rating_counts'] >= m]
    
    
    # R is the average rating of the movie/tv show
    # v is the number of votes for the movie/tv show
    # m is the minimum votes required to be listed in the suggestion
    # C is the mean vote across the whole data
    

    # Funtion that computes the weighted rating (based on IMDb's function/formula)
    def weighted_rating(x, m = m, C = C):
        v = x['rating_counts']
        R = x['rating']
        
        # returning the formula
        return (v / (v + m) * R) + (m / (m + v) * C)
    
    # new column score for our weighted rating
    weighted_ratings_mean['score'] = weighted_ratings_mean.apply(weighted_rating, axis = 1)
    
    # sorting our score by descending order
    weighted_ratings_mean = weighted_ratings_mean.sort_values('score', ascending = False)
    
    # returning the weighted ratings df
    return weighted_ratings_mean.head(30)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)    
    movielens = get_movielens_data()
    
    #movielens recommendations
    movielens_recommendations = simple_recommender(movielens, 'movielens')
    print("MovieLens top 10 recommendations")
    print(movielens_recommendations.head(10))
    print("")
    
    tmdb = get_tmdb_data()
    
    #tmdb recommendations
    tmdb_recommendations = simple_recommender(tmdb, 'tmdb')
    print("TMDB top 10 recommendations")
    print(tmdb_recommendations.head(10))
    print("")
