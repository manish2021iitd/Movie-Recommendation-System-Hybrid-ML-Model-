# CineMatch — Hybrid ML Movie Recommender

A production-grade movie recommendation system combining **content-based filtering** (TF-IDF) and **collaborative filtering** (Matrix Factorization via SVD) in a hybrid engine, served via a cinematic Streamlit UI.

---

## Architecture

<img width="1194" height="785" alt="Screenshot 2026-05-01 at 11 54 20" src="https://github.com/user-attachments/assets/969a04ff-1279-45f9-9e3e-4c7989e28510" />


## Features

| Feature | Detail |
|---|---|
| **Content-Based** | TF-IDF (1-2 grams), weighted metadata soup, cosine similarity |
| **Collaborative** | Truncated SVD matrix factorization, item-item latent similarity |
| **Hybrid Blend** | Configurable α weight + popularity signal |
| **Diversity** | Maximal Marginal Relevance (MMR) re-ranking |
| **Cold-Start** | Profile-based recs from liked movies (no user history needed) |
| **Evaluation** | Precision@K, Recall@K, NDCG@K, Intra-List Diversity |
| **UI** | 5-tab Streamlit app with Plotly charts |

## Project Structure

```
movie_recommender/
├── app.py                    # Streamlit UI (5 tabs)
├── recommender.py            # Pipeline: load → train → expose
├── requirements.txt
├── data/
│   ├── generate_data.py      # Synthetic dataset (50 movies, 500 users)
│   ├── movies.csv            # Generated on first run
│   └── ratings.csv           # Generated on first run
├── models/
│   ├── content_based.py      # TF-IDF + cosine similarity
│   ├── collaborative.py      # SVD matrix factorization
│   └── hybrid.py             # Blending + MMR re-ranking
└── utils/
    └── evaluation.py         # Offline metrics
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The data is auto-generated on first run (50 curated movies, 500 synthetic users, ~13k ratings). To use your own data, replace `movies.csv` and `ratings.csv` with your dataset following the same schema.

## Replacing with Real Data (MovieLens)

```python
# Download MovieLens 100k from https://grouplens.org/datasets/movielens/
# Then adapt column names:
movies_df = pd.read_csv("ml-latest-small/movies.csv")
ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
# Map 'userId' → 'user_id', 'movieId' → 'movie_id', etc.
```

## Evaluation Results (Synthetic Data)

| Metric | Score |
|---|---|
| Precision@10 | ~13% |
| Recall@10 | ~31% |
| NDCG@10 | ~22% |
| Diversity | ~72% |

These are conservative figures on a small catalog. On MovieLens 1M, collaborative filtering typically achieves Precision@10 of 25–35%.

## Improvements Made Over Original Spec

- **Weighted TF-IDF soup**: genres get 3× weight, director/cast get 2×  
- **Profile-based recs**: aggregate content similarity from multiple liked films  
- **MMR diversity re-ranking**: prevents genre/director clustering  
- **Popularity blending**: small 0.1× boost for vote count signal  
- **Offline evaluation suite**: 4 metrics with hold-out split  
- **Cinematic dark UI**: Playfair Display + DM Sans, gold/red palette  
- **Plotly charts**: genre distribution, rating histogram, scatter, radar  
- **5-tab layout**: Similar / Watchlist / Trending / Analytics / Model Info  
