"""
Hybrid Recommender Engine.
Blends content-based and collaborative filtering scores with:
  - Configurable alpha weight
  - Diversity re-ranking (MMR)
  - Popularity boosting / penalizing
  - Cold-start fallback
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    def __init__(self, content_model, cf_model, alpha: float = 0.5):
        """
        alpha=1.0 → pure content-based
        alpha=0.0 → pure collaborative
        """
        self.content_model = content_model
        self.cf_model = cf_model
        self.alpha = alpha
        self.movies_df = content_model.movies_df

    def _normalize(self, series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    def _mmr_rerank(self, candidates: pd.DataFrame, top_n: int, diversity_lambda: float = 0.3) -> pd.DataFrame:
        """
        Maximal Marginal Relevance for diversity.
        Balances relevance score against similarity to already-selected items.
        """
        if len(candidates) <= top_n:
            return candidates

        selected = []
        remaining = list(candidates.index)
        scores = candidates["hybrid_score"].to_dict()
        genres_map = {idx: set(candidates.loc[idx, "genres"].split("|")) for idx in remaining}

        while len(selected) < top_n and remaining:
            if not selected:
                best = max(remaining, key=lambda x: scores[x])
            else:
                def mmr_score(idx):
                    rel = scores[idx]
                    max_sim = max(
                        len(genres_map[idx] & genres_map[s]) / max(len(genres_map[idx] | genres_map[s]), 1)
                        for s in selected
                    )
                    return (1 - diversity_lambda) * rel - diversity_lambda * max_sim
                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return candidates.loc[selected]

    def recommend_for_movie(
        self,
        movie_id: int,
        top_n: int = 8,
        diversity: bool = True,
        popularity_boost: float = 0.1
    ) -> pd.DataFrame:
        """Item-based hybrid: similar movies to a given movie."""
        cb = self.content_model.get_similar_movies(movie_id, top_n=top_n * 3)
        cf = self.cf_model.get_similar_movies(movie_id, top_n=top_n * 3)

        if cb.empty:
            return cf.head(top_n) if not cf.empty else pd.DataFrame()

        merged = cb[["movie_id", "title", "year", "genres", "director",
                      "cast", "description", "imdb_rating", "votes", "content_score"]].copy()

        if not cf.empty:
            cf_scores = cf.set_index("movie_id")["cf_score"]
            merged["cf_score"] = merged["movie_id"].map(cf_scores).fillna(0)
        else:
            merged["cf_score"] = 0

        # Add movies that are in CF but not CB
        if not cf.empty:
            extra = cf[~cf["movie_id"].isin(merged["movie_id"])].copy()
            extra["content_score"] = 0
            extra = extra.rename(columns={"cf_score": "cf_score"})
            merged = pd.concat([merged, extra[merged.columns]], ignore_index=True)

        # Normalize both scores
        merged["content_score_n"] = self._normalize(merged["content_score"])
        merged["cf_score_n"] = self._normalize(merged["cf_score"])

        # Popularity signal (log-normalized votes)
        merged["pop_score"] = np.log1p(merged["votes"])
        merged["pop_score"] = self._normalize(merged["pop_score"])

        # Hybrid blend
        merged["hybrid_score"] = (
            self.alpha * merged["content_score_n"] +
            (1 - self.alpha) * merged["cf_score_n"] +
            popularity_boost * merged["pop_score"]
        )

        merged = merged.sort_values("hybrid_score", ascending=False)
        merged = merged[merged["movie_id"] != movie_id]

        if diversity:
            merged = self._mmr_rerank(merged.head(top_n * 2), top_n)
        else:
            merged = merged.head(top_n)

        merged["recommendation_score"] = self._normalize(merged["hybrid_score"]) * 100
        return merged.reset_index(drop=True)

    def recommend_for_user_profile(
        self,
        liked_movie_ids: list,
        top_n: int = 8,
        diversity: bool = True
    ) -> pd.DataFrame:
        """Profile-based: recommendations from liked movies."""
        cb = self.content_model.get_profile_recommendations(liked_movie_ids, top_n=top_n * 3)

        if cb.empty:
            return pd.DataFrame()

        merged = cb[["movie_id", "title", "year", "genres", "director",
                      "cast", "description", "imdb_rating", "votes", "content_score"]].copy()

        # Try CF for each liked movie and aggregate
        all_cf_scores = {}
        for mid in liked_movie_ids:
            cf = self.cf_model.get_similar_movies(mid, top_n=top_n * 2)
            if not cf.empty:
                for _, row in cf.iterrows():
                    all_cf_scores[row["movie_id"]] = max(
                        all_cf_scores.get(row["movie_id"], 0), row["cf_score"]
                    )

        merged["cf_score"] = merged["movie_id"].map(all_cf_scores).fillna(0)
        merged["content_score_n"] = self._normalize(merged["content_score"])
        merged["cf_score_n"] = self._normalize(merged["cf_score"])
        merged["pop_score"] = self._normalize(np.log1p(merged["votes"]))

        merged["hybrid_score"] = (
            self.alpha * merged["content_score_n"] +
            (1 - self.alpha) * merged["cf_score_n"] +
            0.1 * merged["pop_score"]
        )

        merged = merged[~merged["movie_id"].isin(liked_movie_ids)]
        merged = merged.sort_values("hybrid_score", ascending=False)

        if diversity:
            merged = self._mmr_rerank(merged.head(top_n * 2), top_n)
        else:
            merged = merged.head(top_n)

        merged["recommendation_score"] = self._normalize(merged["hybrid_score"]) * 100
        return merged.reset_index(drop=True)

    def get_genre_recommendations(self, genres: list, top_n: int = 8) -> pd.DataFrame:
        """Get top movies by genre from the catalog."""
        mask = self.movies_df["genres"].apply(
            lambda g: any(genre in g.split("|") for genre in genres)
        )
        result = self.movies_df[mask].copy()
        result["recommendation_score"] = self._normalize(result["imdb_rating"]) * 100
        return result.sort_values("imdb_rating", ascending=False).head(top_n)

    def get_trending(self, top_n: int = 10) -> pd.DataFrame:
        """Return highest-rated + most-voted movies as 'trending'."""
        df = self.movies_df.copy()
        df["score"] = (
            0.6 * self._normalize(df["imdb_rating"]) +
            0.4 * self._normalize(np.log1p(df["votes"]))
        )
        return df.sort_values("score", ascending=False).head(top_n)

    def set_alpha(self, alpha: float):
        self.alpha = np.clip(alpha, 0.0, 1.0)
