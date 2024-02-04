# -*- coding: utf-8 -*-
"""
Atharv Gupta
Offers recommendations based on the content, by looking for similar features 
"""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from read_and_analyze_movie_data_from_file import get_movielens_data
from read_and_analyze_movie_data_from_file import get_tmdb_data


def content_based_recommender(dataset, title, feature):
    # removing duplicate titles from the dataset
    dup_removal = dataset.drop_duplicates(subset=['title'])    
    dup_removal = dup_removal.reset_index()
    
    # filling NA values with blanks
    dup_removal[feature] = dup_removal[feature].fillna('')
    
    # Panda Series for the titles and their indices
    title_indices = pd.Series(dup_removal.index, index=dup_removal['title'])
    
    # importing CountVectorizer and creating the count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(dup_removal[feature])
    # cosine similarity for the distance between them
    cosine_sim = cosine_similarity(count_matrix)
    
    # index of the movie that we select
    index = title_indices[title]
    
    # similarity scores of all the movies
    sim_scores = list(enumerate(cosine_sim[index]))
    
    # removing the movie that we selected to avoid duplicates
    sim_scores.pop(index)
    
    # sorting the scores by descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)    
    
    # list of indices of those movies
    movie_indices = [i[0] for i in sim_scores]
    
    # list of indices of similar movies
    sim_scores_sorted = [i[1] for i in sim_scores]
    
    # similar movies based on cosine similarity score
    similar_movies = pd.DataFrame(sim_scores_sorted, 
                               index=dup_removal['title'].iloc[movie_indices], 
                               columns = ['content_based'])
    
    return similar_movies

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)    
    movielens = get_movielens_data()
    
    #movielens recommendations
    movielens_recommendations = content_based_recommender(movielens, 'Avatar (2009)', 'genres')    
    print("MovieLens top 10 recommendations")
    print(movielens_recommendations.head(10))
    print("")
    
    tmdb = get_tmdb_data()
    
    #tmdb recommendations
    tmdb_recommendations = content_based_recommender(tmdb, 'Superman', 'keywords')
    print("TMDB top 10 recommendations")
    print(tmdb_recommendations.head(10))
    print("")