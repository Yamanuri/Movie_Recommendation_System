
import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.recommender import load_movies, load_ratings, ContentRecommender, ItemItemCF, hybrid_recommend

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="centered")

st.title("🎬 CineMatch — Movie Recommender")
st.write("Content-based, collaborative, and hybrid recommendations. Clean and simple.")

data_dir = Path(__file__).resolve().parents[1] / "data"
movies_csv = data_dir / "movies.csv"
ratings_csv = data_dir / "ratings.csv"

# Load datasets
with st.spinner("Loading data..."):
    movies = load_movies(str(movies_csv))
    content_model = ContentRecommender(movies)

    cf_model = None
    if ratings_csv.exists():
        try:
            ratings = load_ratings(str(ratings_csv))
            cf_model = ItemItemCF(ratings, movies)
        except Exception as e:
            st.warning(f"Collaborative filtering disabled: {e}")

st.subheader("Find Similar Movies")
col1, col2 = st.columns([3,1])
with col1:
    movie_title = st.text_input("Type a movie title", value=str(movies['title'].iloc[0]) if len(movies) else "")
with col2:
    topk = st.number_input("Top K", min_value=1, max_value=50, value=10, step=1)

mode = st.radio("Recommendation mode", ["Content-based", "Collaborative (item-item)", "Hybrid"], horizontal=True)

run = st.button("Recommend")

if run:
    if not movie_title:
        st.error("Please enter a movie title.")
    else:
        try:
            if mode == "Content-based":
                recs = content_model.recommend(movie_title, top_k=int(topk))
            elif mode == "Collaborative (item-item)":
                if cf_model is None:
                    st.error("Collaborative model not available (ratings.csv missing or invalid).")
                    st.stop()
                recs = cf_model.recommend(movie_title, top_k=int(topk))
            else:
                recs = hybrid_recommend(movie_title, content_model, cf_model, top_k=int(topk))

            st.success("Here are your recommendations:")
            st.dataframe(recs)
        except Exception as e:
            st.error(f"Could not generate recommendations: {e}")

st.markdown("---")
st.caption("Built with Streamlit • Easy to explain: data → features → similarity → recommendations.")
