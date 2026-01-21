# Hybrid Movie Recommender System 🎬🎥

## Overview
This project implements a **Hybrid Movie Recommendation System** that integrates **three recommendation models** to provide personalized movie suggestions:  

1. **Content-Based Filtering**  
   - Uses movie metadata (genres) to find similar movies.  
   - Uses **TF-IDF and cosine similarity** to measure movie similarity.  

2. **Collaborative Filtering (User-Based)**  
   - Recommends movies based on user behavior and rating patterns.  
   - Computes **Pearson correlation** between users.  

3. **Matrix Factorization (SVD)**  
   - Learns latent factors between users and movies.  

The system is implemented using **Python**, **Pandas**, **NumPy**, **Scikit-learn**, **Surprise**, and **Streamlit** for the interactive UI.

---

## Features
- Interactive **Streamlit UI** for selecting a movie or user.  
- Generates **top-10 movie recommendations** based on the chosen method.  
- Displays **predicted ratings** and allows exploring content-based, collaborative, or SVD-based recommendations.  
- Evaluates model performance using **RMSE**.

---

## How to Run
1. Clone the repository:  
```bash
git clone <repo-link>
cd <folder-name>


```



2. Install dependencies:  
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:  
```bash
streamlit run app.py


To learn more about the dataset visit: https://grouplens.org/datasets/movielens/100k/