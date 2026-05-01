"""
Central pipeline: loads data, trains models, exposes the hybrid recommender.
"""
import os
import sys
import pandas as pd
import numpy as np

# Make imports work from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data.generate_data import generate_movies_df, generate_ratings_df
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeModel
from models.hybrid import HybridRecommender


def build_recommender(alpha: float = 0.5) -> tuple:
    """Build and return (recommender, movies_df, ratings_df)."""
    # 1. Load / generate data
    movies_path = os.path.join(_HERE, "data", "movies.csv")
    ratings_path = os.path.join(_HERE, "data", "ratings.csv")

    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
    else:
        movies_df = generate_movies_df()
        ratings_df = generate_ratings_df(movies_df)
        os.makedirs(os.path.join(_HERE, "data"), exist_ok=True)
        movies_df.to_csv(movies_path, index=False)
        ratings_df.to_csv(ratings_path, index=False)

    # 2. Fit models
    cb_model = ContentBasedModel()
    cb_model.fit(movies_df)

    cf_model = CollaborativeModel(n_factors=30, n_iter=15)
    cf_model.fit(ratings_df, movies_df)

    # 3. Build hybrid
    recommender = HybridRecommender(cb_model, cf_model, alpha=alpha)

    return recommender, movies_df, ratings_df


if __name__ == "__main__":
    print("Building recommender system...")
    rec, movies, ratings = build_recommender()
    print(f"Loaded {len(movies)} movies, {len(ratings)} ratings")

    # Quick test
    test_movie = movies.iloc[0]
    print(f"\nRecommendations similar to: {test_movie['title']}")
    recs = rec.recommend_for_movie(test_movie["movie_id"], top_n=5)
    for _, r in recs.iterrows():
        print(f"  {r['title']} ({r['year']}) — score: {r['recommendation_score']:.1f}")
