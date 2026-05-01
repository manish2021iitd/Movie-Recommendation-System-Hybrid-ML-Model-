"""
Content-Based Filtering using TF-IDF on movie metadata.
Features: description, genres, director, cast.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            max_features=5000
        )
        self.similarity_matrix = None
        self.movies_df = None
        self.feature_matrix = None

    def _build_soup(self, row):
        """Combine metadata into a weighted textual soup."""
        genres = row["genres"].replace("|", " ")
        cast = row["cast"].replace("|", " ")
        director = row["director"]
        desc = row["description"]

        # Weight: genres x3, director x2, cast x2, description x1
        return f"{genres} {genres} {genres} {director} {director} {cast} {cast} {desc}"

    def fit(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        soup = self.movies_df.apply(self._build_soup, axis=1)
        self.feature_matrix = self.tfidf.fit_transform(soup)
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        return self

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        idx = self.movies_df.index[self.movies_df["movie_id"] == movie_id]
        if len(idx) == 0:
            return pd.DataFrame()
        idx = idx[0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i != idx][:top_n]
        movie_indices = [i for i, _ in scores]
        sim_scores = [s for _, s in scores]
        result = self.movies_df.iloc[movie_indices].copy()
        result["content_score"] = sim_scores
        return result

    def get_profile_recommendations(self, liked_movie_ids: list, top_n: int = 10) -> pd.DataFrame:
        """Recommendations based on multiple liked movies (user profile)."""
        if not liked_movie_ids:
            return pd.DataFrame()

        indices = []
        for mid in liked_movie_ids:
            idx = self.movies_df.index[self.movies_df["movie_id"] == mid]
            if len(idx) > 0:
                indices.append(idx[0])

        if not indices:
            return pd.DataFrame()

        # Aggregate similarity: mean across liked movies
        profile_sim = np.mean(self.similarity_matrix[indices], axis=0)

        scores = list(enumerate(profile_sim))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i not in indices][:top_n]

        movie_indices = [i for i, _ in scores]
        sim_scores = [s for _, s in scores]
        result = self.movies_df.iloc[movie_indices].copy()
        result["content_score"] = sim_scores
        return result

    def get_feature_names(self, top_n: int = 20):
        """Return top TF-IDF feature names for explainability."""
        mean_scores = np.asarray(self.feature_matrix.mean(axis=0)).ravel()
        top_indices = mean_scores.argsort()[-top_n:][::-1]
        return [self.tfidf.get_feature_names_out()[i] for i in top_indices]
