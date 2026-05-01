"""
Collaborative Filtering using Matrix Factorization (Truncated SVD).
Builds a user-item matrix and decomposes it for latent factor learning.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeModel:
    def __init__(self, n_factors: int = 50, n_iter: int = 20):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.svd = TruncatedSVD(n_components=n_factors, n_iter=n_iter, random_state=42)

        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.movies_df = None
        self.ratings_df = None
        self.movie_id_to_idx = {}
        self.user_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df.reset_index(drop=True)

        # Build index maps
        unique_users = ratings_df["user_id"].unique()
        unique_movies = movies_df["movie_id"].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        # Build sparse user-item matrix
        rows, cols, vals = [], [], []
        for _, row in ratings_df.iterrows():
            uid = self.user_id_to_idx.get(row["user_id"])
            mid = self.movie_id_to_idx.get(row["movie_id"])
            if uid is not None and mid is not None:
                rows.append(uid)
                cols.append(mid)
                vals.append(row["rating"])

        self.user_item_matrix = csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_movies)
        )

        # Decompose: user_factors (U * Sigma), item_factors (V^T)
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_  # shape: (n_factors, n_movies)

        # Normalize for cosine similarity
        self.user_factors_norm = normalize(self.user_factors)
        self.item_factors_norm = normalize(self.item_factors.T)  # (n_movies, n_factors)

        # Item-item similarity in latent space
        self.item_similarity = cosine_similarity(self.item_factors_norm)
        return self

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Item-item collaborative similarity."""
        idx = self.movie_id_to_idx.get(movie_id)
        if idx is None:
            return pd.DataFrame()

        sim_scores = self.item_similarity[idx]
        top_indices = np.argsort(sim_scores)[::-1]
        top_indices = [i for i in top_indices if i != idx][:top_n]

        top_movie_ids = [self.idx_to_movie_id[i] for i in top_indices]
        top_scores = [sim_scores[i] for i in top_indices]

        result = self.movies_df[self.movies_df["movie_id"].isin(top_movie_ids)].copy()
        score_map = dict(zip(top_movie_ids, top_scores))
        result["cf_score"] = result["movie_id"].map(score_map)
        return result.sort_values("cf_score", ascending=False)

    def get_user_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Predict ratings for a user and return top-N unseen movies."""
        uid = self.user_id_to_idx.get(user_id)
        if uid is None:
            return pd.DataFrame()

        user_vec = self.user_factors[uid]
        predicted_ratings = user_vec @ self.item_factors

        seen_movie_ids = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["movie_id"]
        )

        scores = []
        for mid, pred in zip(self.idx_to_movie_id.values(), predicted_ratings):
            if mid not in seen_movie_ids:
                scores.append((mid, pred))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        top_movie_ids = [s[0] for s in scores]
        top_scores = [s[1] for s in scores]

        result = self.movies_df[self.movies_df["movie_id"].isin(top_movie_ids)].copy()
        score_map = dict(zip(top_movie_ids, top_scores))
        result["cf_score"] = result["movie_id"].map(score_map)
        return result.sort_values("cf_score", ascending=False)

    def get_explained_variance(self) -> float:
        return float(np.sum(self.svd.explained_variance_ratio_))

    def get_user_stats(self, user_id: int) -> dict:
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        return {
            "n_rated": len(user_ratings),
            "avg_rating": round(user_ratings["rating"].mean(), 2) if len(user_ratings) > 0 else 0,
            "top_genres": self._get_top_genres(user_id)
        }

    def _get_top_genres(self, user_id: int, top_n: int = 3) -> list:
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return []
        high_rated = user_ratings[user_ratings["rating"] >= 4.0]
        rated_movies = self.movies_df[self.movies_df["movie_id"].isin(high_rated["movie_id"])]
        genre_counts = {}
        for genres in rated_movies["genres"]:
            for g in genres.split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        return sorted(genre_counts, key=genre_counts.get, reverse=True)[:top_n]
