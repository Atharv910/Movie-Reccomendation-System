# -*- coding: utf-8 -*-
"""
Atharv Gupta
Read movie dataset from file and analyze
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_movielens_data():   
    ml_movies = pd.read_csv('movielens/movies.csv')
    ml_ratings = pd.read_csv('movielens/ratings.csv')
    
    movielens = pd.merge(ml_ratings, ml_movies, on='movieId')
    
    movielens['genres'] = movielens['genres'].str.replace("|"," ")
    
    return movielens

def analyze_movielens_data(movielens):
    print("MovieLens data")
    print(movielens.head())
    print("")

    # plot of distribution of ratings
    plt.figure(figsize = (12, 8))
    sns.countplot(x = "rating", data = movielens)
    
    plt.title("MovieLens Ratings Distribution")
    plt.xlabel("Ratings")
    plt.ylabel("Number of Ratings (millions)")
    plt.show()
    
    # users with most ratings
    movielens_user_ratings = movielens.groupby("userId")["rating"].count().sort_values(ascending = False)
    print("MovieLens top 10 users with most ratings")
    print(movielens_user_ratings.head(10))
    print("")
    
    # gloval average rating
    movielens_global_rating = movielens['rating'].values.sum() / np.count_nonzero(movielens['rating'].values)
    print('The global average rating for MovieLens dataset is %.2f' % movielens_global_rating)
    print("")
   
    # popular movies based on highest average rating
    movielens_popular_movies = pd.DataFrame(movielens.groupby("title")["rating"].mean())
    # counting how many users rated the title
    movielens_popular_movies['rating_counts'] = pd.DataFrame(movielens.groupby('title')['rating'].count())
    
    # We will use cutoff m as the 90th percentile, so for a movie to be recommended
    # it must have more votes than at least 90% of the movies on the list
    m = movielens_popular_movies['rating_counts'].quantile(0.9)
    
    print("MovieLens top 10 popular movies")
    print(movielens_popular_movies[movielens_popular_movies.rating_counts >= m].sort_values('rating', ascending = False).head(10))
    print("")
    
def get_tmdb_data():   
    tmdb = pd.read_csv('tmdb/tmdb_dataset.csv')
    tmdb = tmdb.rename(columns = {'vote_average' : 'rating', 
                                                'vote_count' : 'rating_counts'})    
    return tmdb

def analyze_tmdb_data(tmdb):
    print("TMDB data")
    print(tmdb.head())
    print("")

    # plot of distribution of ratings
    plt.figure(figsize = (12, 8))
    sns.histplot(tmdb["rating"], bins = 6, kde = False)
    
    plt.title("TMDB Ratings Distribution")
    plt.xlabel("Ratings")
    plt.ylabel("Number of Ratings")
    plt.show()
    
    # users with most ratings
    tmdb_movie_ratings = tmdb.groupby("title")["rating_counts"].sum().sort_values(ascending = False)
    print("TMDB top 10 users with most ratings")
    print(tmdb_movie_ratings.head(10))
    print("")
    
    # gloval average rating
    tmdb_global_rating = tmdb['rating'].values.sum() / np.count_nonzero(tmdb['rating'].values)
    print('The global average rating for TMDB dataset is %.2f' % tmdb_global_rating)
    print("")
   
    # movies with highest popularity score
    tmdb_popular_movies = tmdb[['title', 'popularity']].sort_values('popularity', ascending = False)
    print("TMDB top 10 popular movies")
    print(tmdb_popular_movies.head(10))

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)    
    movielens = get_movielens_data()
    
    #analyze movielens data
    analyze_movielens_data(movielens)
    
    tmdb = get_tmdb_data()
    
    #analyze tmdb data
    analyze_tmdb_data(tmdb)
    