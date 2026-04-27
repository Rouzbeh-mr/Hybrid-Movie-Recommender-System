import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

warnings.filterwarnings('ignore')

nltk.download('wordnet',   quiet=True)
nltk.download('stopwords', quiet=True)

# ── File paths — resolved relative to this script ─────────────────────────────
# Works on Streamlit Cloud (GitHub-connected) and locally.
APP_DIR   = Path(__file__).parent
FILE_NAME = str(APP_DIR / "movie_similarity_matrix.pkl")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align: center;'>
        <h1>🎬 Hybrid Movie Recommendation System</h1>
        <p style='font-size: 16px; color: #666;'>Powered by Rouzbeh</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ONE-TIME DATA LOADING  (cached in session_state)
#
# Actual CSV column names:
#   movies_cleaned.csv : movie_id, title, genres, year, tmdb_id, imdb_id,
#                        overview, tagline, tmdb_rating, tmdb_votes,
#                        imdb_rating, imdb_votes, keywords, cast, director
#   df_cleaned.csv     : userId, movieId, rating, title, genres, year
# ══════════════════════════════════════════════════════════════════════════════
if 'initialised' not in st.session_state:

    # ── Load CSVs ──────────────────────────────────────────────────────────────
    movies_cleaned = pd.read_csv(APP_DIR / 'movies_cleaned.csv', low_memory=False)
    ratings        = pd.read_csv(APP_DIR / 'df_cleaned.csv')

    # ── Rename movie_id -> movieId to share one join key ──────────────────────
    movies_cleaned.rename(columns={'movie_id': 'movieId'}, inplace=True)

    # ── Fill nulls in text columns only ───────────────────────────────────────
    for col in ['genres', 'overview', 'tagline', 'title',
                'keywords', 'cast', 'director']:
        if col in movies_cleaned.columns:
            movies_cleaned[col] = movies_cleaned[col].fillna('')

    # ── Numeric casts ─────────────────────────────────────────────────────────
    for col in ['tmdb_votes', 'imdb_votes', 'tmdb_rating', 'imdb_rating']:
        if col in movies_cleaned.columns:
            movies_cleaned[col] = pd.to_numeric(
                movies_cleaned[col], errors='coerce'
            ).fillna(0)

    # vote_count equivalent for min_ratings filter = tmdb_votes
    movies_cleaned['vote_count'] = movies_cleaned['tmdb_votes'].astype(int)

    # ── Filter users with fewer than 10 ratings ───────────────────────────────
    user_rating_counts = ratings.groupby('userId').size()
    users_to_drop      = user_rating_counts[user_rating_counts < 10].index
    ratings            = ratings[~ratings['userId'].isin(users_to_drop)].copy()

    # ── Build 'body' feature column (same logic as Models.ipynb) ──────────────
    movies_cleaned['genres_text'] = movies_cleaned['genres'].map(
        lambda x: ' '.join(str(x).split('|')).lower()
    )
    text_feature_cols = [
        'overview', 'tagline', 'keywords', 'genres_text', 'director', 'cast'
    ]
    existing = [c for c in text_feature_cols if c in movies_cleaned.columns]
    movies_cleaned['body'] = (
        movies_cleaned[existing]
        .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        .str.lower()
    )
    movies_cleaned.drop(columns=['genres_text'], inplace=True, errors='ignore')

    # ── Text cleaning (same as Models.ipynb clean_text) ───────────────────────
    lemmatizer    = WordNetLemmatizer()
    english_stops = set(stopwords.words('english'))

    def clean_text(text):
        cleaned = []
        for word in str(text).split():
            if word not in english_stops:
                word = lemmatizer.lemmatize(
                    re.sub('[^a-zA-Z0-9]', '', word)
                )
                if word:
                    cleaned.append(word)
        return ' '.join(cleaned)

    movies_cleaned['body'] = movies_cleaned['body'].map(clean_text)

    # ── TF-IDF vectorization ──────────────────────────────────────────────────
    tfidf        = TfidfVectorizer(min_df=1, max_df=0.9)
    tfidf_matrix = tfidf.fit_transform(movies_cleaned['body'])

    # Load pre-computed matrix if available and shape matches
    try:
        content_similarity = pickle.load(open(FILE_NAME, 'rb'))
        if content_similarity.shape[0] != len(movies_cleaned):
            raise ValueError("Shape mismatch")
        similarity_matrix = content_similarity
    except Exception:
        similarity_matrix = cosine_similarity(
            sparse.csr_matrix(tfidf_matrix)
        )

    # Title-indexed content similarity DataFrame (= notebook df_content_sim)
    df_content_sim = pd.DataFrame(
        similarity_matrix,
        index=movies_cleaned['title'].values,
        columns=movies_cleaned['title'].values
    )

    # Shared lookup dict
    movie_id_to_title = movies_cleaned.set_index('movieId')['title'].to_dict()

    # ratings_with_titles: df_cleaned already has 'title' column
    ratings_with_titles = ratings.copy()

    # ── User-item matrix, mean-centring, cosine similarity ────────────────────
    user_item       = ratings_with_titles.pivot_table(
        values='rating', index='userId', columns='title'
    )
    norm_user_item  = user_item.subtract(user_item.mean(axis=1), axis='rows')
    user_similarity = cosine_similarity(
        sparse.csr_matrix(norm_user_item.fillna(0))
    )
    df_user_sim = pd.DataFrame(
        user_similarity,
        index=user_item.index,
        columns=user_item.index
    )

    # ── Store in session_state ────────────────────────────────────────────────
    st.session_state.movies_cleaned      = movies_cleaned
    st.session_state.df_content_sim      = df_content_sim
    st.session_state.movie_id_to_title   = movie_id_to_title
    st.session_state.ratings_with_titles = ratings_with_titles
    st.session_state.norm_user_item      = norm_user_item
    st.session_state.df_user_sim         = df_user_sim
    st.session_state.df_user_original    = ratings_with_titles.copy()
    st.session_state.df_user_current     = ratings_with_titles.copy()
    st.session_state.initialised         = True

# Ensure UI keys always exist
for key, default in [
    ('recommendations_shown', False),
    ('new_userId', None),
    ('num_movies', 3),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Pull shared objects ───────────────────────────────────────────────────────
movies_cleaned    = st.session_state.movies_cleaned
df_content_sim    = st.session_state.df_content_sim
movie_id_to_title = st.session_state.movie_id_to_title
norm_user_item    = st.session_state.norm_user_item
df_user_sim       = st.session_state.df_user_sim


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS  (exact logic from Models.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def get_content_similar_movies(user_id, top_n=20, min_ratings_per_movie=5):
    """Content-based recommendations — exact logic from Models.ipynb."""
    user_ratings = st.session_state.df_user_current[
        st.session_state.df_user_current['userId'] == user_id
    ]
    if user_ratings.empty:
        return pd.DataFrame()

    watched_titles = user_ratings['title'].values

    eligible_titles = set(
        movies_cleaned[
            movies_cleaned['vote_count'] >= min_ratings_per_movie
        ]['title'].unique()
    )

    mean_rating  = user_ratings['rating'].mean()
    liked_titles = user_ratings.loc[
        user_ratings['rating'] >= mean_rating, 'title'
    ].values

    score_series_list = []
    for title in liked_titles:
        if title in df_content_sim.columns:
            scores = df_content_sim[title].drop(
                labels=watched_titles, errors='ignore'
            )
            scores = scores[scores.index.isin(eligible_titles)]
            score_series_list.append(scores)

    if not score_series_list:
        return pd.DataFrame()

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
        movies_cleaned[['title', 'genres']], combined,
        on='title', how='inner'
    )
    return (
        result.sort_values('content_similarity', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_collaborative_similar_movies(
    user_id, similarity_threshold=0.1, top_n=20, min_ratings_per_movie=5
):
    """User-based collaborative filtering — exact logic from Models.ipynb."""
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
        norm_ui.loc[norm_ui.index.isin(similar_users.index)]
        .dropna(axis=1, how='all')
    )

    candidate_movies = [
        m for m in neighbour_movies.columns
        if m not in target_watched and m in eligible_titles
    ]

    movie_scores = {}
    for movie in candidate_movies:
        numerator = denominator = 0.0
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
        movies_cleaned[['title', 'genres']], score_df,
        on='title', how='inner'
    )
    return result.sort_values('predicted_rating', ascending=False).reset_index(drop=True)


def get_hybrid_similar_movies(
    user_id, similarity_threshold=0.1, top_n=20, min_ratings_per_movie=5
):
    """Hybrid recommendations — exact logic from Models.ipynb."""
    content_df = get_content_similar_movies(
        user_id, top_n=top_n * 5,
        min_ratings_per_movie=min_ratings_per_movie
    )
    collab_df = get_collaborative_similar_movies(
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
        content_df['content_similarity'] = _minmax(
            content_df['content_similarity']
        )
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
        "Tell us what you think about these movies to get "
        "personalised recommendations:"
    )

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
            cur = len(st.session_state.user_ratings)
            if number > cur:
                for _ in range(cur, number):
                    st.session_state.user_ratings.append(("", 3.0))
            else:
                st.session_state.user_ratings = (
                    st.session_state.user_ratings[:number]
                )
        st.rerun()

    options = movies_cleaned['title'].values.tolist()

    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = [("", 3.0) for _ in range(number)]

    st.info(
        "💡 **Tip:** You can type directly in the dropdown "
        "box to search for movies!"
    )

    for i in range(number):
        st.markdown(f"**Movie {i+1}**")
        col1, col2 = st.columns([3, 1])

        with col1:
            prev_movie = (
                st.session_state.user_ratings[i][0]
                if i < len(st.session_state.user_ratings) else ""
            )
            idx = (
                options.index(prev_movie)
                if prev_movie and prev_movie in options else 0
            )
            movie = st.selectbox(
                'Movie title', key=f"movie_{i}",
                options=options, index=idx,
                help="Start typing to search for a movie"
            )

        with col2:
            prev_rating = (
                st.session_state.user_ratings[i][1]
                if i < len(st.session_state.user_ratings) else 3.0
            )
            rating = st.select_slider(
                'Rating', key=f"rating_{i}",
                options=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                value=prev_rating
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
        valid = True
        for mv, _ in st.session_state.user_ratings:
            if not mv:
                valid = False
                st.error(
                    "Please select a movie for all entries "
                    "before getting recommendations."
                )
                break

        if valid:
            new_userId = int(
                st.session_state.df_user_original['userId'].max()
            ) + 1
            st.session_state.new_userId = new_userId

            new_rows = []
            for mv, rat in st.session_state.user_ratings:
                row = movies_cleaned[movies_cleaned['title'] == mv]
                if row.empty:
                    continue
                new_rows.append({
                    'userId':  new_userId,
                    'movieId': int(row['movieId'].values[0]),
                    'rating':  rat,
                    'title':   mv,
                    'genres':  row['genres'].values[0],
                    'year':    row['year'].values[0]
                    if 'year' in row.columns else '',
                })

            df_new = pd.DataFrame(new_rows).drop_duplicates()
            st.session_state.df_user_current = pd.concat(
                [st.session_state.df_user_original, df_new],
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
            disp = hybrid_recs[['title', 'genres', 'hybrid_score']].copy()
            disp.index = disp.index + 1
            disp.columns = ['Movie Title', 'Genres', 'Hybrid Score']
            disp['Hybrid Score'] = disp['Hybrid Score'].round(4)
            st.dataframe(disp, use_container_width=True)

    with tab2:
        st.markdown("#### 📄 Content-Based Recommendations (Top 20)")
        with st.spinner("Generating content-based recommendations..."):
            content_recs = get_content_similar_movies(
                user_id, top_n=20, min_ratings_per_movie=5
            )
        if content_recs.empty:
            st.warning(
                "No content-based recommendations could be generated."
            )
        else:
            disp = content_recs[
                ['title', 'genres', 'content_similarity']
            ].copy()
            disp.index = disp.index + 1
            disp.columns = ['Movie Title', 'Genres', 'Content Similarity']
            disp['Content Similarity'] = disp['Content Similarity'].round(4)
            st.dataframe(disp, use_container_width=True)

    with tab3:
        st.markdown(
            "#### 👥 Collaborative Filtering Recommendations (Top 20)"
        )
        with st.spinner("Generating collaborative recommendations..."):
            collab_recs = get_collaborative_similar_movies(
                user_id, similarity_threshold=0.1,
                top_n=20, min_ratings_per_movie=5
            )
        if collab_recs.empty:
            st.warning(
                "No collaborative recommendations found. "
                "This may happen if no similar users exist "
                "for your ratings."
            )
        else:
            disp = collab_recs[
                ['title', 'genres', 'predicted_rating']
            ].copy()
            disp.index = disp.index + 1
            disp.columns = ['Movie Title', 'Genres', 'Predicted Rating']
            disp['Predicted Rating'] = disp['Predicted Rating'].round(4)
            st.dataframe(disp, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 14px; color: #666;'>"
        "🎬 Powered by Hybrid Recommendation Engine | "
        "Built with Streamlit</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            '🔄 Start Over with New Ratings', use_container_width=True
        ):
            st.session_state.recommendations_shown = False
            st.rerun()
