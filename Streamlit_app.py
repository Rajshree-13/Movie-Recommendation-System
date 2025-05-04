import streamlit as st
import pandas as pd
import pickle

books_df = pd.read_csv('Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings_df = pd.read_csv('Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

books_df.columns = ['book_id', 'book_title', 'author', 'year', 'publisher']
ratings_df.columns = ['user_id', 'book_id', 'rating']

with open('book_recommender_model.pkl', 'rb') as file:
    model = pickle.load(file)


def get_recommendation(user_id, num_recommendations=5):
    rated_books = ratings_df[ratings_df['user_id'] == user_id]['book_id'].unique()
    all_books = books_df['book_id'].unique()
    books_to_predict = list(set(all_books) - set(rated_books))

    predictions = [model.predict(user_id, book_id) for book_id in books_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_books = predictions[:num_recommendations]

    book_ids = [pred.iid for pred in top_books]
    return books_df[books_df['book_id'].isin(book_ids)][['book_title', 'author', 'publisher']]


st.title("Book Recommendation System")

user_id = st.number_input("Enter your User ID:", min_value=1, step=1)
num_recommendations = st.slider("Number of books to recommend:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    if user_id in ratings_df['user_id'].unique():
        st.success(f"Top {num_recommendations} Recommendations for User ID {user_id}:")
        recommendations = get_recommendation(user_id, num_recommendations)
        st.table(recommendations)
    else:
        st.error("User ID not found. Please enter a valid User ID.")