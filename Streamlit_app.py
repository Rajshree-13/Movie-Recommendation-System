import streamlit as st
import pandas as pd
import pickle

movies_df = pd.read_csv('movies.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings_df = pd.read_csv('ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

movies_df.columns = ['movieId', 'title', 'genres']
ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']

with open('book_recommender_model.pkl', 'rb') as file:
    model = pickle.load(file)


def get_recommendation(userId, num_recommendations=5):
    rated_movie = ratings_df[ratings_df['userId'] == userId]['movieId'].unique()
    all_movie = movie_df['movieId'].unique()
    movie_to_predict = list(set(all_movie) - set(rated_movie))

    predictions = [model.predict(userId, movieId) for movieId in books_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_books = predictions[:num_recommendations]

    movie_ids = [pred.iid for pred in top_books]
    return movie_df[movie_df['movieId'].isin(movie_ids)][['movieId', 'title', 'genres']]


st.title("Book Recommendation System")

userId = st.number_input("Enter your User ID:", min_value=1, step=1)
num_recommendations = st.slider("Number of books to recommend:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    if userId in ratings_df['userId'].unique():
        st.success(f"Top {num_recommendations} Recommendations for User ID {userId}:")
        recommendations = get_recommendation(userId, num_recommendations)
        st.table(recommendations)
    else:
        st.error("User ID not found. Please enter a valid User ID.")
