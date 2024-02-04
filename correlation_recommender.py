# -*- coding: utf-8 -*-
"""
Atharv Gupta
Offers recommendations based on the correlation between the ratings of the
movies
"""
import pandas as pd
from read_and_analyze_movie_data_from_file import get_movielens_data
import warnings

warnings.filterwarnings('ignore')

def correlation_recommender(dataset, title):    
    # pivot table by userId with movies as columns and rating as values
    movie_rating_pivot = dataset.pivot_table(index='userId', columns='title', 
                                              values='rating')
    
    # specific movie and the user's rating for it
    user_rating = movie_rating_pivot[title]
    
    # correlation between other movies
    similar_movies_corr = movie_rating_pivot.corrwith(user_rating)
    
    # creating a dataframe from the correlations and dropping NA values
    similar_movies = pd.DataFrame(similar_movies_corr, columns=['correlation'])
    similar_movies.dropna(inplace=True)
    
    # fiding the mean of our rating by title
    ratings_mean = pd.DataFrame(dataset.groupby('title')['rating'].mean())
    
    # counting how many users rated the title
    ratings_mean['rating_counts'] = pd.DataFrame(dataset.groupby('title')['rating'].count())
    
    # We will use cutoff m as the 90th percentile, so for a movie to be recommended
    # it must have more votes than at least 90% of the movies on the list
    m = ratings_mean['rating_counts'].quantile(0.9)
    
    # adding rating_counts column
    similar_movies = similar_movies.join(ratings_mean['rating_counts'])
    
    # filtering the similar movies based on the min requirement of rating_counts
    similar_movies = similar_movies[similar_movies['rating_counts'] 
                                    > m].sort_values('correlation', ascending=False)
    
    # returning the similar movies df
    return similar_movies.head(30)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)    
    movielens = get_movielens_data()
    
    # TMDB doesn't have individual ratings of users so only doing
    # movielens recommendations
    movielens_recommendations = correlation_recommender(movielens, 'Shawshank Redemption, The (1994)')
    print("MovieLens correlation recommendations")
    print(movielens_recommendations.head(10))