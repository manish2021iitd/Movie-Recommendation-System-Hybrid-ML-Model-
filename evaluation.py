"""
Evaluation metrics for the recommendation system.
Implements: Precision@K, Recall@K, NDCG@K, Coverage, Diversity.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant items found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, mid in enumerate(recommended_ids[:k]):
        if mid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)

    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(recommendations: pd.DataFrame) -> float:
    """Average genre dissimilarity between recommended items."""
    if len(recommendations) < 2:
        return 0.0

    genre_sets = [set(g.split("|")) for g in recommendations["genres"]]
    dissim_scores = []

    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            a, b = genre_sets[i], genre_sets[j]
            jaccard = len(a & b) / max(len(a | b), 1)
            dissim_scores.append(1 - jaccard)

    return np.mean(dissim_scores)


def catalog_coverage(all_recommended_ids: list, total_movies: int) -> float:
    """Fraction of the catalog that has been recommended at least once."""
    return len(set(all_recommended_ids)) / total_movies


def evaluate_hybrid(recommender, ratings_df: pd.DataFrame, k: int = 10, sample_users: int = 50):
    """
    Hold-out evaluation: for each user, hide 20% of their ratings as ground truth,
    then measure recommendation quality.
    """
    results = {"precision": [], "recall": [], "ndcg": [], "diversity": []}

    user_ids = ratings_df["user_id"].unique()
    if len(user_ids) > sample_users:
        user_ids = np.random.choice(user_ids, sample_users, replace=False)

    for uid in user_ids:
        user_ratings = ratings_df[ratings_df["user_id"] == uid]
        if len(user_ratings) < 5:
            continue

        train, test = train_test_split(user_ratings, test_size=0.2, random_state=42)

        # Use highly-rated train movies as profile
        liked_train = train[train["rating"] >= 4.0]["movie_id"].tolist()
        if not liked_train:
            continue

        # Ground truth: highly-rated test movies
        relevant = set(test[test["rating"] >= 4.0]["movie_id"])
        if not relevant:
            continue

        recs = recommender.recommend_for_user_profile(liked_train, top_n=k, diversity=False)
        if recs.empty:
            continue

        rec_ids = recs["movie_id"].tolist()
        results["precision"].append(precision_at_k(rec_ids, relevant, k))
        results["recall"].append(recall_at_k(rec_ids, relevant, k))
        results["ndcg"].append(ndcg_at_k(rec_ids, relevant, k))
        results["diversity"].append(intra_list_diversity(recs))

    return {
        metric: round(np.mean(vals) * 100, 2) if vals else 0.0
        for metric, vals in results.items()
    }
