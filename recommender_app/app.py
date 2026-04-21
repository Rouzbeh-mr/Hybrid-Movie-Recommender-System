from hashlib import new
from ssl import Options
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from functions import *
import os
import gdown

FILE_ID = "19vF_vGJV-tLu09t0aHCN03Sg1ZxmvWIg"
FILE_NAME = "movie_similarity_matrix.pkl"

if not os.path.exists(FILE_NAME):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, FILE_NAME, quiet=False)
    
st.header('Personalized Movie Recommendations')

# Initialize session state
if 'recommendations_shown' not in st.session_state:
    st.session_state.recommendations_shown = False
if 'new_userId' not in st.session_state:
    st.session_state.new_userId = None
if 'df_user_original' not in st.session_state:
    # Load original data only once
    df_content = pd.read_csv('clean_content.csv')
    df_user = pd.read_csv('ratings_title.csv')
    df_user.rename(columns={'userId':'user_id', 'movieId':'movie_id'}, inplace=True)
    st.session_state.df_content = df_content
    st.session_state.df_user_original = df_user.copy()
    st.session_state.df_user_current = df_user.copy()
    
    content_similarity = pickle.load(open('movie_similarity_matrix.pkl', 'rb'))
    st.session_state.df_content_sim = pd.DataFrame(content_similarity, index=df_content['title'].values, columns=df_content['title'].values)

# Use session state variables
df_content = st.session_state.df_content
df_user = st.session_state.df_user_current
df_content_sim = st.session_state.df_content_sim

#Get data from the user
if not st.session_state.recommendations_shown:
    new_user_data = []
    number = int(st.number_input('How many movies would you like to rate?', min_value=3, value=3, step=1))
    
    options = df_content['title'].values.tolist()
    
    # Store ratings in session state
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = []
    
    for i in range(number):
        col1, col2 = st.columns(2)
        with col1:
            movie = st.selectbox(
                'Movie title',
                key=f"movie_{i}",
                options=options
            )
        with col2:
            rating = st.select_slider(
                'Rate the movie',
                options=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                key=f"rating_{i}",
                value=3.0
            )
        
        # Store temporarily
        if len(st.session_state.user_ratings) <= i:
            st.session_state.user_ratings.append((movie, rating))
        else:
            st.session_state.user_ratings[i] = (movie, rating)
    
    if st.button('Get Recommendations'):
        # Use stored ratings
        new_user_data = st.session_state.user_ratings
        
        #Add new_user_data to user database
        new_userId = st.session_state.df_user_original['user_id'].sort_values().values[-1] + 1
        st.session_state.new_userId = new_userId
        
        new_user = []
        for movie, rating in new_user_data:
            new_ratings = {}
            new_ratings['user_id'] = new_userId
            new_ratings['rating'] = rating
            new_ratings['movie_id'] = df_content.loc[df_content['title'] == movie, 'movie_id'].values[0]
            new_ratings['title'] = movie
            new_ratings['genres'] = df_content.loc[df_content['title'] == movie, 'genres'].values[0]
            new_ratings['year'] = df_content[df_content['title'] == movie]['year'].values[0]
            new_user.append(new_ratings)

        df_new_user = pd.DataFrame(new_user).drop_duplicates()

        #Add the new user to the df_user dataframe
        st.session_state.df_user_current = pd.concat([st.session_state.df_user_original, df_new_user])
        df_user = st.session_state.df_user_current
        
        st.session_state.recommendations_shown = True
        st.rerun()

# Show recommendations if button was clicked
if st.session_state.recommendations_shown and st.session_state.new_userId is not None:
    
    # Recompute matrices with the updated user data
    df_user = st.session_state.df_user_current
    
    #Create User-Item Matrix
    user_item = df_user.pivot_table(values='rating', index='user_id', columns='title')
    
    #Normalize User-Item matrix
    norm_user_item = user_item.subtract(user_item.mean(axis=1), axis='rows')
    
    #User-User similarity matrix
    user_similarity = cosine_similarity(sparse.csr_matrix(norm_user_item.fillna(0)))
    df_user_sim = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)
    
    def get_content_similar_movies(user):
    #Current/target user
    df_current_user = df_user[df_user['user_id'] == user]

    #Movies watched by the current/target user
    user_watched_movies = df_current_user['title'].values
    
    # If no movies watched, return empty
    if len(user_watched_movies) == 0:
        return pd.DataFrame(columns=['title', 'genres', 'content_similarity'])

    #User's mean rating
    user_mean_rating = df_current_user['rating'].mean()

    #Filter the list of movies by like/dislike based on user's rating
    user_movies = []
    for movie in user_watched_movies:
        # Get the rating for this specific movie and compare (fix the broadcasting issue)
        movie_rating = df_current_user[df_current_user['title'] == movie]['rating'].values
        
        # Check if array is not empty and compare the first element
        if len(movie_rating) > 0 and movie_rating[0] >= user_mean_rating:
            user_movies.append(movie)
    
    # If no movies above mean rating, use all rated movies instead
    if len(user_movies) == 0:
        user_movies = user_watched_movies.tolist()
        
    #Create an empty dataframe to store movie recommendations
    similar_movies = pd.DataFrame()
    #Loop through each movie seen by the user
    for movie in user_movies:
        #Add similarity score for each movie with user_movie
        #Remove movies that the user has already seen
        if movie in df_content_sim.index:
            try:
                # Get similarity scores for this movie
                movie_similarities = df_content_sim[movie]
                
                # Drop movies the user has already seen
                movies_to_drop = [m for m in user_watched_movies if m in movie_similarities.index]
                if movies_to_drop:
                    movie_similarities = movie_similarities.drop(movies_to_drop, errors='ignore')
                
                # Add to dataframe
                similar_movies = pd.concat([similar_movies, movie_similarities.to_frame().T])
            except Exception as e:
                # If dropping fails, just use all similar movies
                similar_movies = pd.concat([similar_movies, df_content_sim[movie].to_frame().T])
    
    if similar_movies.empty:
        # Return popular movies as fallback
        popular_movies = df_content.nlargest(20, 'imdb_rating')[['title', 'genres']].copy()
        popular_movies['content_similarity'] = 0.5
        return popular_movies
    
    #Add the similarity score of each movie
    # Sum across all rows (movies rated by user)
    content_rec = pd.DataFrame(similar_movies.sum()).reset_index().rename(columns={'index': 'title',
                    0: 'content_similarity'})
    
    # Remove any NaN values
    content_rec = content_rec.dropna(subset=['content_similarity'])
    
    if content_rec.empty:
        popular_movies = df_content.nlargest(20, 'imdb_rating')[['title', 'genres']].copy()
        popular_movies['content_similarity'] = 0.5
        return popular_movies
    
    # Apply min-max normalization
    if len(content_rec) > 0 and content_rec['content_similarity'].max() != content_rec['content_similarity'].min():
        content_rec['content_similarity'] = (content_rec['content_similarity'] - content_rec['content_similarity'].min()) / (content_rec['content_similarity'].max() - content_rec['content_similarity'].min())
    elif len(content_rec) > 0:
        content_rec['content_similarity'] = 0.5
    
    result = pd.merge(df_content[['title', 'genres']], content_rec, how='inner').sort_values(by='content_similarity', ascending=False)
    
    if result.empty:
        popular_movies = df_content.nlargest(20, 'imdb_rating')[['title', 'genres']].copy()
        popular_movies['content_similarity'] = 0.5
        return popular_movies
        
    return result

    def get_user_similar_movies(user, similarity_threshold):
        #Extract similar users
        if user not in df_user_sim.index:
            # Fallback: return popular movies
            popular_movies = df_content.nlargest(20, 'imdb_rating')[['title', 'genres', 'year']].copy()
            popular_movies['user_similarity'] = 0.5
            return popular_movies
            
        similar_users = df_user_sim[df_user_sim[user] > similarity_threshold][user].sort_values(ascending=False)[1:]
        
        # If no similar users found, lower the threshold
        if similar_users.empty and similarity_threshold > 0:
            similar_users = df_user_sim[df_user_sim[user] > 0][user].sort_values(ascending=False)[1:5]
        
        # If still no similar users, return popular movies
        if similar_users.empty:
            popular_movies = df_content.nlargest(20, 'imdb_rating')[['title', 'genres', 'year']].copy()
            popular_movies['user_similarity'] = 0.5
            return popular_movies
        
        #Extract movies watched by the target user
        target_user_movies = norm_user_item[norm_user_item.index == user].dropna(axis=1, how='all')
        
        #Extract movies watched by similar users
        similar_user_movies = norm_user_item[norm_user_item.index.isin(similar_users.index)].dropna(axis=1, how='all')
        
        #Remove movies watched by target user
        for column in target_user_movies.columns:
            if column in similar_user_movies.columns:
                similar_user_movies.drop(column, axis=1, inplace=True)
        
        if similar_user_movies.empty:
            # Return popular movies not watched by user
            watched_movies = target_user_movies.columns.tolist()
            popular_movies = df_content[~df_content['title'].isin(watched_movies)].nlargest(20, 'imdb_rating')[['title', 'genres', 'year']].copy()
            popular_movies['user_similarity'] = 0.5
            return popular_movies
        
        #Weighted average
        movie_score = {}
        for movie in similar_user_movies.columns:
            movie_rating = similar_user_movies[movie]
            numerator = 0
            denominator = 0
            for sim_user in similar_users.index:
                if sim_user in movie_rating.index and pd.notnull(movie_rating[sim_user]):
                    weighted_score = similar_users[sim_user] * movie_rating[sim_user]
                    numerator += weighted_score
                    denominator += similar_users[sim_user]
            if denominator > 0:
                movie_score[movie] = numerator / denominator
        
        if not movie_score:
            watched_movies = target_user_movies.columns.tolist()
            popular_movies = df_content[~df_content['title'].isin(watched_movies)].nlargest(20, 'imdb_rating')[['title', 'genres', 'year']].copy()
            popular_movies['user_similarity'] = 0.5
            return popular_movies
        
        movie_score_df = pd.DataFrame(movie_score.items(), columns=['title', 'user_similarity'])
        
        # Apply min-max normalization
        if len(movie_score_df) > 0 and movie_score_df['user_similarity'].max() != movie_score_df['user_similarity'].min():
            movie_score_df['user_similarity'] = (movie_score_df['user_similarity'] - movie_score_df['user_similarity'].min()) / (movie_score_df['user_similarity'].max() - movie_score_df['user_similarity'].min())
        elif len(movie_score_df) > 0:
            movie_score_df['user_similarity'] = 0.5
        
        user_rec = pd.merge(df_content[['title', 'genres', 'year']], movie_score_df[['title', 'user_similarity']], how='inner')
        
        if user_rec.empty:
            popular_movies = df_content.nlargest(20, 'imdb_rating')[['title', 'genres', 'year']].copy()
            popular_movies['user_similarity'] = 0.5
            return popular_movies
            
        return user_rec.sort_values(by=['user_similarity', 'year'], ascending=False)

    def hybrid_recommender(user):
        # Always get both content and user-based recommendations
        content_df = get_content_similar_movies(user)
        user_df = get_user_similar_movies(user, 0.1)
        
        # If both are empty (shouldn't happen with fallbacks), return popular movies
        if content_df.empty and user_df.empty:
            popular_movies = df_content.nlargest(10, 'imdb_rating')[['title', 'genres', 'imdb_rating']].copy()
            popular_movies['Similarity Score'] = 0.5
            popular_movies.rename(columns={'title': 'Movie Title', 'imdb_rating': 'IMDb Rating'}, inplace=True)
            popular_movies.insert(0, 'Rank', range(1, len(popular_movies) + 1))
            return popular_movies
        
        # If one is empty, use the other
        if content_df.empty:
            user_df['similarity_score'] = user_df['user_similarity']
            top_scores = user_df.sort_values(by='similarity_score', ascending=False)[:10]
        elif user_df.empty:
            content_df['similarity_score'] = content_df['content_similarity']
            top_scores = content_df.sort_values(by='similarity_score', ascending=False)[:10]
        else:
            # Merge both recommendation sources
            try:
                content_user_scores = pd.merge(content_df, user_df, on=['title', 'genres'])
                
                if content_user_scores.empty:
                    # If no common movies, combine both lists
                    content_df['similarity_score'] = content_df['content_similarity']
                    user_df['similarity_score'] = user_df['user_similarity']
                    combined = pd.concat([content_df[['title', 'genres', 'similarity_score']], 
                                         user_df[['title', 'genres', 'similarity_score']]], 
                                         ignore_index=True)
                    combined = combined.drop_duplicates(subset=['title'])
                    top_scores = combined.sort_values(by='similarity_score', ascending=False)[:10]
                else:
                    # Both scores are normalized to [0,1] before averaging
                    content_user_scores['similarity_score'] = (content_user_scores['content_similarity'] + content_user_scores['user_similarity']) / 2
                    top_scores = content_user_scores.sort_values(by='similarity_score', ascending=False)[:10]
            except Exception as e:
                # Fallback to content-based only
                content_df['similarity_score'] = content_df['content_similarity']
                top_scores = content_df.sort_values(by='similarity_score', ascending=False)[:10]
        
        # Merge with movie details
        recommendations = pd.merge(df_content[['title', 'genres', 'imdb_rating']], 
                                   top_scores[['title', 'similarity_score']], 
                                   on='title')
        
        recommendations.rename(columns={'title': 'Movie Title', 
                                        'imdb_rating': 'IMDb Rating', 
                                        'similarity_score': 'Similarity Score'}, inplace=True)
        
        recommendations = recommendations.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)
        recommendations.insert(0, 'Rank', range(1, len(recommendations) + 1))
        
        return recommendations
    
    # Display recommendations
    recommendations_df = hybrid_recommender(st.session_state.new_userId)
    st.table(recommendations_df)
    
    # Add button to get new recommendations with different movies
    if st.button('Start Over with New Ratings'):
        # Clear session state to reset
        for key in ['recommendations_shown', 'new_userId', 'user_ratings']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.df_user_current = st.session_state.df_user_original.copy()
        st.rerun()
