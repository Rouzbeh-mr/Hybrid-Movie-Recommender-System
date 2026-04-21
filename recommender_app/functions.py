import numpy as np
import pandas as pd


def get_content_similar_movies(user):
    
    #Current/target user
    df_current_user = df_user[df_user['user_id'] == user]

    #Movies watched by the current/target user
    user_watched_movies = df_current_user['title'].values

    #User's mean rating
    user_mean_rating = df_current_user['rating'].mean()

    #Filter the list of movies by like/dislike based on user's rating
    user_movies = []
    for movie in user_watched_movies:
        if df_current_user[df_current_user['title'] == movie]['rating'].values >= user_mean_rating:
            user_movies.append(movie)
        
    #Create an empty dataframe to store movie recommendations for each movie seen by the user
    similar_movies = pd.DataFrame()
    #Loop through each movie seen by the user
    for movie in user_movies:
        #Add similarity score for each movie with user_movie
        #Remove movies that the user has already seen
        similar_movies = pd.concat([similar_movies, df_content_sim[movie].drop(user_watched_movies).to_frame().T])
    #Add the similarity score of each movie and select the movies with high scores
    content_rec = pd.DataFrame(similar_movies.sum()).reset_index().rename(columns={'index': 'title',
                    0: 'content_similarity'})
    
    # Apply min-max normalization to content_similarity
    if len(content_rec) > 0 and content_rec['content_similarity'].max() != content_rec['content_similarity'].min():
        content_rec['content_similarity'] = (content_rec['content_similarity'] - content_rec['content_similarity'].min()) / (content_rec['content_similarity'].max() - content_rec['content_similarity'].min())
    elif len(content_rec) > 0:
        content_rec['content_similarity'] = 0.5  # Set to mid value if all scores are equal
    
    return pd.merge(df_content[['title', 'genres']], content_rec, how='inner').sort_values(by='content_similarity', ascending=False)

def get_user_similar_movies(user, similarity_threshold):
    
    #Extract similar users and their similarity score with the target user
    similar_users = df_user_sim[df_user_sim[user] > similarity_threshold][user].sort_values(ascending=False)[1:]

    #Extract movies watched by the target user and their score with the target user
    target_user_movies = norm_user_item[norm_user_item == user].dropna(axis =1, how= 'all')

    #Extract movies watched by similar users and their score with the similar users
    similar_user_movies = norm_user_item[norm_user_item.index.isin(similar_users.index)].dropna(axis=1, how = 'all')

    #Keep the movies watched by similar users but not by the target user: 
    for column in target_user_movies.columns: 
        if column in similar_user_movies.columns:
            similar_user_movies.drop(column, axis=1, inplace=True)
        
    #Weighted average
    movie_score = {}
    #Loop through the movies seen by similar users
    for movie in similar_user_movies.columns:
        #Extract the rating for each movie
        movie_rating = similar_user_movies[movie]
        #Variable to calculate numerator of the weighted average
        #This must be calculated for each movie
        numerator = 0
        #Variable to calculate the denominator of the weighted average
        denominator = 0
        #Loop through the similar users for that movie
        for user in similar_users.index:
            #If the similar user has seen the movie
            if pd.notnull(movie_rating[user]):
                #Weighted score is the product of user similarity score and movie rating by the similar user
                weighted_score = similar_users[user] * movie_rating[user]
                numerator += weighted_score
                denominator += similar_users[user]
        movie_score[movie] = numerator / denominator
    #Save the movie and the similarity score in a dataframe
    movie_score = pd.DataFrame(movie_score.items(), columns=['title', 'user_similarity'])
    
    # Apply min-max normalization to user_similarity
    if len(movie_score) > 0 and movie_score['user_similarity'].max() != movie_score['user_similarity'].min():
        movie_score['user_similarity'] = (movie_score['user_similarity'] - movie_score['user_similarity'].min()) / (movie_score['user_similarity'].max() - movie_score['user_similarity'].min())
    elif len(movie_score) > 0:
        movie_score['user_similarity'] = 0.5  # Set to mid value if all scores are equal
    
    user_rec = pd.merge(df_content[['title','genres','year']], movie_score[['title', 'user_similarity']], how='inner')
    return user_rec.sort_values(by=['user_similarity', 'year'], ascending=False)

def hybrid_recommender(user):
    content_user_scores = pd.merge(get_content_similar_movies(user), get_user_similar_movies(user, 0.1))
    
    # Both scores are now normalized to [0,1] before averaging
    content_user_scores['similarity_score'] = (content_user_scores['content_similarity'] + content_user_scores['user_similarity']) / 2
    
    top_scores = content_user_scores.sort_values(by='similarity_score', ascending=False)[:10]
    
    recommendations = pd.merge(df_content[['title','genres','imdb_rating']], top_scores[['title','similarity_score']], on='title')
    recommendations.rename(columns={'title':'Movie Title', 'imdb_rating': 'IMDb Rating', 'similarity_score':'Similarity Score'}, inplace=True)
    recommendations = recommendations.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)
    recommendations.insert(0, 'Rank', range(1, len(recommendations) + 1))

    return recommendations

    return recommendations
