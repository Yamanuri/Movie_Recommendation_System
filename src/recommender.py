
"""
CineMatch - Simple, clean movie recommender
-------------------------------------------
Content-based (TF-IDF on genres/overview) + optional item-item collaborative filtering.
Keep it lightweight and easy to explain.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Data loading utilities
# ----------------------
def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to standardize column names: title, genres, overview, etc.
    cols = {c.lower(): c for c in df.columns}
    # Normalize title
    if "title" not in cols:
        # Try some alternates
        for alt in ["movie_title", "name", "original_title"]:
            if alt in cols:
                df.rename(columns={cols[alt]: "title"}, inplace=True)
                break
    # Normalize genres
    if "genres" not in cols:
        for alt in ["genre", "listed_in"]:
            if alt in cols:
                df.rename(columns={cols[alt]: "genres"}, inplace=True)
                break
    # Normalize overview/description
    if "overview" not in cols:
        for alt in ["description", "plot", "tagline", "summary"]:
            if alt in cols:
                df.rename(columns={cols[alt]: "overview"}, inplace=True)
                break
    # Fill missing fields used in content model
    if "genres" not in df.columns:
        df["genres"] = ""
    if "overview" not in df.columns:
        df["overview"] = ""
    # Ensure title exists
    if "title" not in df.columns:
        # Fall back to index as title to avoid crashes
        df["title"] = df.index.astype(str)
    return df

def load_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize expected columns
    # We expect: userId, movieId/title, rating
    lower = {c.lower(): c for c in df.columns}
    # Common variations
    if "userid" not in lower:
        for alt in ["user_id", "user"]:
            if alt in lower:
                df.rename(columns={lower[alt]: "userId"}, inplace=True)
                break
    if "movieid" not in lower and "title" not in lower:
        for alt in ["movie_id", "movie", "item_id"]:
            if alt in lower:
                df.rename(columns={lower[alt]: "movieId"}, inplace=True)
                break
    if "rating" not in lower:
        for alt in ["score", "ratings"]:
            if alt in lower:
                df.rename(columns={lower[alt]: "rating"}, inplace=True)
                break
    return df

# ----------------------
# Content-based model
# ----------------------
class ContentRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.copy()
        # Build a single text field
        self.movies["__text"] = (
            self.movies.get("genres", "").astype(str) + " " +
            self.movies.get("overview", "").astype(str)
        ).fillna("")
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
        self.tfidf = self.vectorizer.fit_transform(self.movies["__text"])
        self.title_index = {t.lower(): i for i, t in enumerate(self.movies["title"].astype(str))}

    def recommend(self, title: str, top_k: int = 10) -> pd.DataFrame:
        if not self.title_index:
            raise ValueError("No titles available in movies data.")
        idx = self._find_index_by_title(title)
        if idx is None:
            # fallback: fuzzy find by containment
            idx = self._search_first(title)
            if idx is None:
                raise ValueError(f"Title '{title}' not found in dataset.")
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).ravel()
        # Exclude the same movie
        sims[idx] = -1.0
        top_idx = np.argsort(-sims)[:top_k]
        out = self.movies.iloc[top_idx][["title"]].copy()
        out["similarity"] = sims[top_idx]
        return out.reset_index(drop=True)

        # Helper functions
    def _find_index_by_title(self, title: str) -> Optional[int]:
        return self.title_index.get(str(title).lower())

    def _search_first(self, query: str) -> Optional[int]:
        q = str(query).lower()
        for i, t in enumerate(self.movies["title"].astype(str)):
            if q in t.lower():
                return i
        return None

# ----------------------
# Item-item collaborative filtering (simple, implicit)
# ----------------------
class ItemItemCF:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        # Try to map movieId -> title if both exist; else, infer using title in ratings if present.
        if "movieId" in self.ratings.columns and "movieId" in self.movies.columns:
            self.movie_titles = self.movies.set_index("movieId")["title"].to_dict()
            pivot_index = "movieId"
        elif "title" in self.ratings.columns:
            # Ratings already contain title
            self.movie_titles = {t: t for t in self.ratings["title"].astype(str).unique()}
            pivot_index = "title"
        else:
            # If nothing usable, degrade gracefully by using an enumerated id
            self.ratings["title"] = self.ratings.get("movieId", self.ratings.index).astype(str)
            self.movie_titles = {t: t for t in self.ratings["title"].unique()}
            pivot_index = "title"

        # Build item-user matrix
        user_col = "userId" if "userId" in self.ratings.columns else "user"
        rating_col = "rating" if "rating" in self.ratings.columns else self.ratings.select_dtypes(include=[np.number]).columns[-1]
        mat = self.ratings.pivot_table(index=pivot_index, columns=user_col, values=rating_col, aggfunc="mean").fillna(0.0)
        self.items_index = list(mat.index)
        # Cosine similarity between items
        self.sim = cosine_similarity(mat.values)
        # Map for lookup
        self.index_map = {k: i for i, k in enumerate(self.items_index)}

    def recommend(self, item_key, top_k: int = 10) -> pd.DataFrame:
        # item_key can be movieId or title based on pivot_index used
        if item_key not in self.index_map:
            # try title search (case-insensitive)
            key = self._search_key(item_key)
            if key is None:
                raise ValueError(f"Movie '{item_key}' not found in ratings matrix.")
            item_key = key
        idx = self.index_map[item_key]
        sims = self.sim[idx].copy()
        sims[idx] = -1.0
        top_idx = np.argsort(-sims)[:top_k]
        keys = [self.items_index[i] for i in top_idx]
        titles = [self.movie_titles.get(k, str(k)) for k in keys]
        out = pd.DataFrame({"title": titles, "similarity": sims[top_idx]})
        return out.reset_index(drop=True)

    def _search_key(self, query: str):
        q = str(query).lower()
        for k in self.index_map.keys():
            if q in str(k).lower():
                return k
        return None

# ----------------------
# Simple Hybrid (average ranks of content + CF when both available)
# ----------------------
def hybrid_recommend(title: str, content: ContentRecommender, cf: Optional[ItemItemCF], top_k: int = 10) -> pd.DataFrame:
    base = content.recommend(title, top_k=50)
    if cf is None:
        return base.head(top_k)

    # Try CF by matching title to pivot key; we search in both title and potential ids
    try:
        cf_rec = cf.recommend(title, top_k=50)
    except Exception:
        # If CF can't find by title, just return content-based
        return base.head(top_k)

    # Merge by title and rank-average
    base = base.copy()
    base['rank_c'] = np.arange(len(base))
    cf_rec = cf_rec.copy()
    cf_rec['rank_cf'] = np.arange(len(cf_rec))

    merged = pd.merge(base[['title','rank_c']], cf_rec[['title','rank_cf']], on='title', how='outer')
    merged['rank_c'] = merged['rank_c'].fillna(merged['rank_c'].max() + 10)
    merged['rank_cf'] = merged['rank_cf'].fillna(merged['rank_cf'].max() + 10)
    merged['score'] = - ( (merged['rank_c'].rank(method='min') + merged['rank_cf'].rank(method='min')) / 2.0 )
    merged = merged.sort_values('score', ascending=False).head(top_k)

    # Attach similarity-like placeholder (not meaningful across models, just for display)
    merged = merged[['title']].reset_index(drop=True)
    return merged
