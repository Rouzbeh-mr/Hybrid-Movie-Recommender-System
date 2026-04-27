import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import os
import gdown
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

warnings.filterwarnings('ignore')

# ── NLTK downloads ─────────────────────────────────────────────────────────────
nltk.download('wordnet',   quiet=True)
nltk.download('stopwords', quiet=True)

# ── Download similarity matrix from Drive if not present ──────────────────────
FILE_ID   = "19vF_vGJV-tLu09t0aHCN03Sg1ZxmvWIg"
FILE_NAME = "movie_similarity_matrix.pkl"

if not os.path.exists(FILE_NAME):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, FILE_NAME, quiet=False)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align: center;'>
        <h1>🎬 Hybrid Movie Recommendation System</h1>
        <p style='font-size: 16px; color: #666;'>Powered by Rouzbeh</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Session-state initialisation ──────────────────────────────────────────────
if 'recommendations_shown' not in st.session_state:
    st.session_state.recommendations_shown = False
if 'new_userId' not in st.session_state:
    st.session_state.new_userId = None

if 'df_user_original' not in st.session_state:

    # ── Load CSVs ──────────────────────────────────────────────────────────────
    movies_cleaned = pd.read_csv('movies_cleaned.csv', low_memory=False)
    ratings        = pd.read_csv('df_cleaned.csv')

    # ── Fill nulls needed for modelling ───────────────────────────────────────
    cols_to_fill = [
        'genres', 'original_language', 'original_title', 'overview',
        'popularity', 'tagline', 'title', 'vote_average', 'vote_count',
        'keywords', 'cast', 'director', 'production_companies',
        'production_countries'
    ]
    for col in cols_to_fill:
        if col in movies_cleaned.columns:
            movies_cleaned[col] = movies_cleaned[col].fillna('')

    # ── Numeric casts ─────────────────────────────────────────────────────────
    movies_cleaned['vote_average'] = pd.to_numeric(
        movies_cleaned['vote_average'], errors='coerce').fillna(0).astype(int)
    movies_cleaned['vote_count']   = pd.to_numeric(
        movies_cleaned['vote_count'],   errors='coerce').fillna(0).astype(int)

    # ── Filter users with fewer than 10 ratings ───────────────────────────────
    user_id_col  = 'userId'  if 'userId'  in ratings.columns else 'user_id'
    movie_id_col = 'movieId' if 'movieId' in ratings.columns else 'movie_id'

    user_rating_counts = ratings.groupby(user_id_col).size()
    users_to_drop = user_rating_counts[user_rating_counts < 10].index
    ratings = ratings[~ratings[user_id_col].isin(users_to_drop)].copy()

    # Standardise column names to userId / movieId for the rest of the app
    ratings.rename(
        columns={user_id_col: 'userId', movie_id_col: 'movieId'},
        inplace=True
    )

    # ── Build 'body' feature column (same as notebook) ────────────────────────
    movies_cleaned['genres_text'] = movies_cleaned['genres'].map(
        lambda x: ' '.join(str(x).split('|')).lower()
    )
    text_cols = [
        'overview', 'tagline', 'keywords', 'production_companies',
        'production_countries', 'original_language', 'genres_text',
        'director', 'cast'
    ]
    existing_text_cols = [c for c in text_cols if c in movies_cleaned.columns]
    movies_cleaned['body'] = (
        movies_cleaned[existing_text_cols]
        .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        .str.lower()
    )
    movies_cleaned.drop(
        columns=[c for c in existing_text_cols if c != 'genres_text'],
        inplace=True, errors='ignore'
    )
    if 'genres_text' in movies_cleaned.columns:
        movies_cleaned.drop(columns=['genres_text'], inplace=True, errors='ignore')

    # ── Text cleaning (same as notebook) ──────────────────────────────────────
    lemmatizer    = WordNetLemmatizer()
    english_stops = set(stopwords.words('english'))

    def clean_text(text):
        cleaned = []
        for word in str(text).split():
            if word not in english_stops:
                word = lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '', word))
                if word:
                    cleaned.append(word)
        return ' '.join(cleaned)

    movies_cleaned['body'] = movies_cleaned['body'].map(clean_text)

    # ── TF-IDF + cosine similarity matrix ─────────────────────────────────────
    tfidf        = TfidfVectorizer(min_df=1, max_df=0.9)
    tfidf_matrix = tfidf.fit_transform(movies_cleaned['body'])

    # Try loading pre-computed similarity matrix first; build if unavailable
    try:
        content_similarity = pickle.load(open(FILE_NAME, 'rb'))
        similarity_matrix  = content_similarity
    except Exception:
        similarity_matrix = cosine_similarity(sparse.csr_matrix(tfidf_matrix))

    # Title-indexed similarity DataFrame (same as notebook's df_content_sim)
    df_content_sim = pd.DataFrame(
        similarity_matrix,
        index=movies_cleaned['title'].values,
        columns=movies_cleaned['title'].values
    )

    # ── Shared lookup dicts (same as notebook) ────────────────────────────────
    movie_id_to_title = movies_cleaned.set_index('movieId')['title'].to_dict()

    # ── ratings_with_titles (same as notebook) ────────────────────────────────
    ratings_with_titles = pd.merge(
        ratings,
        movies_cleaned[['movieId', 'title']],
        on='movieId',
        how='left'
    ).drop(columns=['timestamp'], errors='ignore')

    # ── User-item matrix + similarity (same as notebook) ─────────────────────
    user_item      = ratings_with_titles.pivot_table(
        values='rating', index='userId', columns='title'
    )
    norm_user_item = user_item.subtract(user_item.mean(axis=1), axis='rows')
    user_similarity = cosine_similarity(
        sparse.csr_matrix(norm_user_item.fillna(0))
    )
    df_user_sim = pd.DataFrame(
        user_similarity,
        index=user_item.index,
        columns=user_item.index
    )

    # ── Store everything in session state ─────────────────────────────────────
    st.session_state.movies_cleaned      = movies_cleaned
    st.session_state.df_content_sim      = df_content_sim
    st.session_state.movie_id_to_title   = movie_id_to_title
    st.session_state.ratings_with_titles = ratings_with_titles
    st.session_state.user_item           = user_item
    st.session_state.norm_user_item      = norm_user_item
    st.session_state.df_user_sim         = df_user_sim
    st.session_state.df_user_original    = ratings_with_titles.copy()
    st.session_state.df_user_current     = ratings_with_titles.copy()

# ── Pull shared objects from session state ─────────────────────────────────────
movies_cleaned      = st.session_state.movies_cleaned
df_content_sim      = st.session_state.df_content_sim
movie_id_to_title   = st.session_state.movie_id_to_title
ratings_with_titles = st.session_state.ratings_with_titles
norm_user_item      = st.session_state.norm_user_item
df_user_sim         = st.session_state.df_user_sim


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS  (exact logic from Models.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def get_content_similar_movies(user_id, top_n=20, min_ratings_per_movie=5):
    """
    Content-based recommendations — exact logic from Models.ipynb.
    """
    user_ratings = st.session_state.df_user_current[
        st.session_state.df_user_current['userId'] == user_id
    ]

    if user_ratings.empty:
        return pd.DataFrame()

    watched_titles = user_ratings['title'].values

    # Eligible movies
    eligible_movie_data = movies_cleaned[
        movies_cleaned['vote_count'] >= min_ratings_per_movie
    ]
    eligible_titles = set(eligible_movie_data['title'].unique())

    # Personalised mean rating threshold
    mean_rating  = user_ratings['rating'].mean()
    liked_titles = user_ratings.loc[
        user_ratings['rating'] >= mean_rating, 'title'
    ].values

    score_series_list = []
    for title in liked_titles:
        if title in df_content_sim.columns:
            scores = df_content_sim[title].drop(labels=watched_titles, errors='ignore')
            scores = scores[scores.index.isin(eligible_titles)]
            score_series_list.append(scores)

    if not score_series_list:
        return pd.DataFrame()

    # Union index — same as notebook
    union_idx = pd.Index([])
    for s in score_series_list:
        union_idx = union_idx.union(s.index)

    combined = (
        pd.concat(
            [s.reindex(union_idx, fill_value=0) for s in score_series_list],
            axis=1
        )
        .sum(axis=1)
        .reset_index()
        .rename(columns={'index': 'title', 0: 'content_similarity'})
    )

    result = pd.merge(
        movies_cleaned[['title', 'genres']], combined, on='title', how='inner'
    )
    return (
        result.sort_values('content_similarity', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_collaborative_similar_movies(
    user_id, similarity_threshold=0.1, top_n=20, min_ratings_per_movie=5
):
    """
    User-based collaborative filtering — exact logic from Models.ipynb.
    Uses the live df_user_sim that includes the new user.
    """
    # Rebuild user-item matrices to include any newly added user
    current_ratings = st.session_state.df_user_current
    ui      = current_ratings.pivot_table(
        values='rating', index='userId', columns='title'
    )
    norm_ui = ui.subtract(ui.mean(axis=1), axis='rows')
    u_sim   = cosine_similarity(sparse.csr_matrix(norm_ui.fillna(0)))
    df_sim  = pd.DataFrame(u_sim, index=ui.index, columns=ui.index)

    if user_id not in df_sim.index:
        return pd.DataFrame()

    eligible_titles = set(
        movies_cleaned[
            movies_cleaned['vote_count'] >= min_ratings_per_movie
        ]['title'].unique()
    )

    similar_users = (
        df_sim[user_id]
        .drop(index=user_id, errors='ignore')
        .loc[lambda s: s > similarity_threshold]
        .sort_values(ascending=False)
    )

    if similar_users.empty:
        return pd.DataFrame()

    target_watched = set(
        norm_ui.loc[user_id].dropna().index
        if user_id in norm_ui.index else []
    )

    neighbour_movies = (
        norm_ui
        .loc[norm_ui.index.isin(similar_users.index)]
        .dropna(axis=1, how='all')
    )

    candidate_movies = [
        m for m in neighbour_movies.columns
        if m not in target_watched and m in eligible_titles
    ]

    movie_scores = {}
    for movie in candidate_movies:
        numerator   = 0.0
        denominator = 0.0
        for neighbour_id, sim_score in similar_users.items():
            rating = (
                neighbour_movies.loc[neighbour_id, movie]
                if neighbour_id in neighbour_movies.index
                else np.nan
            )
            if pd.notna(rating):
                numerator   += sim_score * rating
                denominator += abs(sim_score)
        if denominator > 0:
            movie_scores[movie] = numerator / denominator

    if not movie_scores:
        return pd.DataFrame()

    score_df = (
        pd.Series(movie_scores)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'index': 'title', 0: 'predicted_rating'})
    )

    result = pd.merge(
        movies_cleaned[['title', 'genres']], score_df, on='title', how='inner'
    )
    return result.sort_values('predicted_rating', ascending=False).reset_index(drop=True)


def get_hybrid_similar_movies(
    user_id, similarity_threshold=0.1, top_n=20, min_ratings_per_movie=5
):
    """
    Hybrid recommendations — exact logic from Models.ipynb.
    Fetches expanded pools, min-max normalises each, then averages.
    """
    content_df = get_content_similar_movies(
        user_id, top_n=top_n * 5,
        min_ratings_per_movie=min_ratings_per_movie
    )
    collab_df  = get_collaborative_similar_movies(
        user_id, similarity_threshold,
        top_n=top_n * 5,
        min_ratings_per_movie=min_ratings_per_movie
    )

    if content_df.empty and collab_df.empty:
        return pd.DataFrame()

    def _minmax(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series

    if not content_df.empty:
        content_df = content_df.copy()
        content_df['content_similarity'] = _minmax(content_df['content_similarity'])

    if not collab_df.empty:
        collab_df = collab_df.copy()
        collab_df['predicted_rating'] = _minmax(collab_df['predicted_rating'])

    hybrid = pd.merge(
        content_df[['title', 'genres', 'content_similarity']]
        if not content_df.empty
        else pd.DataFrame(columns=['title', 'genres', 'content_similarity']),

        collab_df[['title', 'predicted_rating']]
        if not collab_df.empty
        else pd.DataFrame(columns=['title', 'predicted_rating']),

        on='title', how='outer'
    )

    hybrid['content_similarity'] = hybrid['content_similarity'].fillna(0)
    hybrid['predicted_rating']   = hybrid['predicted_rating'].fillna(0)

    # Recover genres for collab-only rows
    if hybrid['genres'].isna().any():
        genre_map = movies_cleaned.set_index('title')['genres'].to_dict()
        hybrid['genres'] = hybrid.apply(
            lambda r: genre_map.get(r['title'], '')
            if pd.isna(r['genres']) else r['genres'],
            axis=1
        )

    hybrid['hybrid_score'] = (
        hybrid['content_similarity'] + hybrid['predicted_rating']
    ) / 2

    return (
        hybrid.sort_values('hybrid_score', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# UI — Rating input
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.recommendations_shown:
    st.markdown("### 📝 Rate Some Movies")
    st.markdown(
        "Tell us what you think about these movies to get personalised recommendations:"
    )

    if 'num_movies' not in st.session_state:
        st.session_state.num_movies = 3

    number = int(
        st.number_input(
            'How many movies would you like to rate?',
            min_value=3,
            value=st.session_state.num_movies,
            step=1
        )
    )

    if number != st.session_state.num_movies:
        st.session_state.num_movies = number
        if 'user_ratings' in st.session_state:
            current_len = len(st.session_state.user_ratings)
            if number > current_len:
                for i in range(current_len, number):
                    st.session_state.user_ratings.append(("", 3.0))
            elif number < current_len:
                st.session_state.user_ratings = st.session_state.user_ratings[:number]
        st.rerun()

    options = movies_cleaned['title'].values.tolist()

    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = [("", 3.0) for _ in range(number)]

    st.info("💡 **Tip:** You can type directly in the dropdown box to search for movies!")

    for i in range(number):
        st.markdown(f"**Movie {i+1}**")
        col1, col2 = st.columns([3, 1])

        with col1:
            previous_movie = (
                st.session_state.user_ratings[i][0]
                if i < len(st.session_state.user_ratings) else ""
            )
            index = options.index(previous_movie) if (
                previous_movie and previous_movie in options
            ) else 0

            movie = st.selectbox(
                'Movie title',
                key=f"movie_{i}",
                options=options,
                index=index,
                help="Start typing to search for a movie"
            )

        with col2:
            previous_rating = (
                st.session_state.user_ratings[i][1]
                if i < len(st.session_state.user_ratings) else 3.0
            )
            rating = st.select_slider(
                'Rating',
                options=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                key=f"rating_{i}",
                value=previous_rating
            )

        if len(st.session_state.user_ratings) <= i:
            st.session_state.user_ratings.append((movie, rating))
        else:
            st.session_state.user_ratings[i] = (movie, rating)

        st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        get_rec_button = st.button(
            '🎯 Get Recommendations', use_container_width=True
        )

    if get_rec_button:
        valid_ratings = True
        for movie, rating in st.session_state.user_ratings:
            if not movie or movie == "":
                valid_ratings = False
                st.error(
                    "Please select a movie for all entries before getting recommendations."
                )
                break

        if valid_ratings:
            new_user_data = st.session_state.user_ratings

            # Assign new userId
            new_userId = (
                st.session_state.df_user_original['userId']
                .sort_values().values[-1] + 1
            )
            st.session_state.new_userId = new_userId

            new_user = []
            for movie, rating in new_user_data:
                movie_row = movies_cleaned[movies_cleaned['title'] == movie]
                if movie_row.empty:
                    continue
                new_user.append({
                    'userId':  new_userId,
                    'movieId': movie_row['movieId'].values[0],
                    'rating':  rating,
                    'title':   movie,
                    'genres':  movie_row['genres'].values[0],
                })

            df_new_user = pd.DataFrame(new_user).drop_duplicates()

            # Append new user to current ratings
            st.session_state.df_user_current = pd.concat(
                [st.session_state.df_user_original, df_new_user],
                ignore_index=True
            )

            st.session_state.recommendations_shown = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# UI — Recommendations display
# ══════════════════════════════════════════════════════════════════════════════
if (
    st.session_state.recommendations_shown
    and st.session_state.new_userId is not None
):
    user_id = st.session_state.new_userId

    st.markdown("### 🎯 Your Personalised Recommendations")

    tab1, tab2, tab3 = st.tabs(
        ["🔀 Hybrid", "📄 Content-Based", "👥 Collaborative"]
    )

    with tab1:
        st.markdown("#### 🔀 Hybrid Recommendations (Top 20)")
        with st.spinner("Generating hybrid recommendations..."):
            hybrid_recs = get_hybrid_similar_movies(
                user_id, similarity_threshold=0.1,
                top_n=20, min_ratings_per_movie=5
            )
        if hybrid_recs.empty:
            st.warning("No hybrid recommendations could be generated.")
        else:
            display_df = hybrid_recs[['title', 'genres', 'hybrid_score']].copy()
            display_df.index = display_df.index + 1
            display_df.columns = ['Movie Title', 'Genres', 'Hybrid Score']
            display_df['Hybrid Score'] = display_df['Hybrid Score'].round(4)
            st.dataframe(display_df, use_container_width=True)

    with tab2:
        st.markdown("#### 📄 Content-Based Recommendations (Top 20)")
        with st.spinner("Generating content-based recommendations..."):
            content_recs = get_content_similar_movies(
                user_id, top_n=20, min_ratings_per_movie=5
            )
        if content_recs.empty:
            st.warning("No content-based recommendations could be generated.")
        else:
            display_df = content_recs[['title', 'genres', 'content_similarity']].copy()
            display_df.index = display_df.index + 1
            display_df.columns = ['Movie Title', 'Genres', 'Content Similarity']
            display_df['Content Similarity'] = display_df['Content Similarity'].round(4)
            st.dataframe(display_df, use_container_width=True)

    with tab3:
        st.markdown("#### 👥 Collaborative Filtering Recommendations (Top 20)")
        with st.spinner("Generating collaborative recommendations..."):
            collab_recs = get_collaborative_similar_movies(
                user_id, similarity_threshold=0.1,
                top_n=20, min_ratings_per_movie=5
            )
        if collab_recs.empty:
            st.warning(
                "No collaborative recommendations found. "
                "This may happen if no similar users exist for your ratings."
            )
        else:
            display_df = collab_recs[['title', 'genres', 'predicted_rating']].copy()
            display_df.index = display_df.index + 1
            display_df.columns = ['Movie Title', 'Genres', 'Predicted Rating']
            display_df['Predicted Rating'] = display_df['Predicted Rating'].round(4)
            st.dataframe(display_df, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 14px; color: #666;'>"
        "🎬 Powered by Hybrid Recommendation Engine | Built with Streamlit"
        "</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔄 Start Over with New Ratings', use_container_width=True):
            st.session_state.recommendations_shown = False
            st.rerun()
