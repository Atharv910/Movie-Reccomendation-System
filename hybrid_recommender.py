# -*- coding: utf-8 -*-
"""
Atharv Gupta
Combines Content-Based and Collaborative Filtering Recommender
"""
import pandas as pd
from read_and_analyze_movie_data_from_file import get_movielens_data
from content_based_recommender import content_based_recommender
from model_based_collaborative_filtering_recommender import collaborative_filtering_recommender

def hybrid_recommender(dataset, title):
    # content based
    content = content_based_recommender(dataset, title, 'genres')
    # collaborative filtering
    collaborative_filtering = collaborative_filtering_recommender(dataset, title)
    
    # joining the dataframes
    hybrid = content.join(collaborative_filtering, how = 'outer')
    
    # score for the hybrid recommender
    hybrid['hybrid'] = (hybrid['content_based'] + hybrid['collaborative']) / 2
    
    # sorting the values by hybrid score
    hybrid = hybrid.sort_values('hybrid', ascending=False)    
    
    # returning the df
    return hybrid.head(30)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)    
    movielens = get_movielens_data()
    
    # TMDB doesn't have individual ratings of users so only doing
    #movielens recommendations
    movielens_recommendations  = hybrid_recommender(movielens, 'Amazing Spider-Man, The (2012)')
    print("MovieLens top 10 recommendations")
    print(movielens_recommendations.head(10))
    print("")