import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# 2a. Load saved model/data
# -------------------------------
movie_similarity = pickle.load(open("movie_similarity.pkl", "rb"))
user_similarity = pickle.load(open("user_similarity.pkl", "rb"))
user_item_matrix_train = pickle.load(open("user_item_matrix_train.pkl", "rb"))
user_factors = pickle.load(open("user_factors.pkl", "rb"))
item_factors = pickle.load(open("item_factors.pkl", "rb"))

# Movie titles
movies = pd.read_csv("u.item", sep="|", encoding="latin-1", 
                     names=['movie_id','title','release_date','video_release_date','IMDb_URL',
                            'unknown','Action','Adventure','Animation',"Children's",'Comedy',
                            'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
                            'Mystery','Romance','Sci-Fi','Thriller','War','Western'])
movie_titles = movies.set_index('movie_id')['title'].to_dict()

# -------------------------------
# 2b. Recommendation functions
# -------------------------------

def recommend_content_based(user_rated_movies, k=10):
    sim_scores = movie_similarity[user_rated_movies].mean(axis=1)
    sim_scores = sim_scores.drop(user_rated_movies, errors='ignore')
    top_k = sim_scores.sort_values(ascending=False).head(k)
    return [(movie_titles[mid], score) for mid, score in top_k.items()]

def recommend_user_cf(user_id, k=10):
    if user_id not in user_item_matrix_train.index:
        return []
    sim_scores = user_similarity[user_id].fillna(0).drop(user_id, errors='ignore')
    sim_scores = sim_scores[sim_scores>0]
    if sim_scores.sum() == 0:
        return []
    weighted_sum = np.dot(user_item_matrix_train.loc[sim_scores.index].fillna(0).T, sim_scores)
    predicted_ratings = pd.Series(weighted_sum / sim_scores.sum(), index=user_item_matrix_train.columns)
    rated_movies = user_item_matrix_train.loc[user_id].dropna().index
    predicted_ratings = predicted_ratings.drop(rated_movies, errors='ignore')
    top_k = predicted_ratings.sort_values(ascending=False).head(k)
    return [(movie_titles[mid], score) for mid, score in top_k.items()]

def recommend_mf(user_id, k=10):
    if user_id not in user_item_matrix_train.index:
        return []
    user_idx = user_item_matrix_train.index.get_loc(user_id)
    pred_ratings = np.dot(user_factors[user_idx], item_factors)
    predicted_ratings = pd.Series(pred_ratings, index=user_item_matrix_train.columns)
    rated_movies = user_item_matrix_train.loc[user_id].dropna().index
    predicted_ratings = predicted_ratings.drop(rated_movies, errors='ignore')
    top_k = predicted_ratings.sort_values(ascending=False).head(k)
    return [(movie_titles[mid], score) for mid, score in top_k.items()]

# -------------------------------
# 3️⃣ Streamlit UI
# -------------------------------
st.title("Movie Recommendation System 🎬")

st.sidebar.header("User Options")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=user_item_matrix_train.index.max(), step=1)
method = st.sidebar.selectbox("Select Recommendation Method", ["Content-Based", "User-CF", "Matrix Factorization"])
k = st.sidebar.slider("Number of Recommendations (K)", min_value=5, max_value=20, value=10)

st.write(f"Recommendations for User {user_id} using {method}")

if st.button("Recommend"):
    if method=="Content-Based":
        # For content-based, pick movies the user has already rated
        rated_movies = user_item_matrix_train.loc[user_id].dropna().index.tolist()
        recs = recommend_content_based(rated_movies, k)
    elif method=="User-CF":
        recs = recommend_user_cf(user_id, k)
    else:
        recs = recommend_mf(user_id, k)
    
    if not recs:
        st.write("No recommendations available for this user.")
    else:
        for title, score in recs:
            st.write(f"{title}  —  Score: {score:.2f}")
