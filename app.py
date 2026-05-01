"""
Movie Recommendation System – Streamlit UI
Hybrid ML recommender with content-based + collaborative filtering.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold: #e8b84b;
    --dark: #0d0d0f;
    --card: #161618;
    --accent: #c94040;
    --text: #e8e4de;
    --muted: #6b6b6b;
    --border: #2a2a2a;
}

/* Global */
html, body, .stApp { background-color: var(--dark) !important; color: var(--text) !important; }
.stApp { font-family: 'DM Sans', sans-serif; }

/* Header */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 300;
}

/* Movie cards */
.movie-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.1rem 0.9rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--accent));
}
.movie-card:hover { border-color: var(--gold); }

.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.15rem;
}
.movie-meta {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.movie-desc {
    font-size: 0.82rem;
    color: #b0a89c;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}
.genre-badge {
    display: inline-block;
    background: rgba(232,184,75,0.12);
    border: 1px solid rgba(232,184,75,0.3);
    color: var(--gold);
    font-size: 0.7rem;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
    font-weight: 500;
}
.score-bar-wrap { margin-top: 0.5rem; }
.score-label { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.2rem; }
.score-bar-bg {
    background: #1e1e20;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--gold));
}
.imdb-badge {
    background: #f5c518;
    color: #000;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
}
.star-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem; }

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: var(--gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* Metric tiles */
.metric-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--gold);
    font-weight: 700;
}
.metric-lbl { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0a0c !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label { color: var(--muted) !important; font-size: 0.8rem !important; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #a03030) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stMetricValue"] { color: var(--gold) !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; }

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load / Cache Models ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from recommender import build_recommender
    return build_recommender(alpha=0.5)


@st.cache_resource(show_spinner=False)
def run_evaluation(_recommender, _ratings_df):
    from utils.evaluation import evaluate_hybrid
    return evaluate_hybrid(_recommender, _ratings_df, k=10, sample_users=80)


# ─── Helper Functions ────────────────────────────────────────────────────────
def render_movie_card(row, show_score=True, rank=None):
    genres_html = "".join(f'<span class="genre-badge">{g}</span>' for g in row["genres"].split("|"))
    score = row.get("recommendation_score", 0)
    rank_html = f'<span style="color:var(--gold);font-size:0.75rem;font-weight:700;">#{rank} </span>' if rank else ''

    card_html = f"""
    <div class="movie-card">
      <div class="movie-title">{rank_html}{row['title']}</div>
      <div class="movie-meta">🎬 {row['director']} &nbsp;·&nbsp; {row['year']} &nbsp;·&nbsp; {row['cast'].replace('|', ', ')}</div>
      <div class="movie-desc">{row['description'][:160]}{'…' if len(row['description']) > 160 else ''}</div>
      <div>{genres_html}</div>
      <div class="star-row">
        <span class="imdb-badge">IMDb</span>
        <span style="color:#f5c518;font-size:0.9rem;">{'★' * int(round(row['imdb_rating']/2))}</span>
        <span style="color:var(--muted);font-size:0.8rem;">{row['imdb_rating']}/10 &nbsp;·&nbsp; {row['votes']:,} votes</span>
      </div>
      {'<div class="score-bar-wrap"><div class="score-label">Match score</div><div class="score-bar-bg"><div class="score-bar-fill" style="width:' + str(min(100, score)) + '%;"></div></div></div>' if show_score else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def genre_distribution_chart(movies_df):
    genre_counts = {}
    for genres in movies_df["genres"]:
        for g in genres.split("|"):
            genre_counts[g] = genre_counts.get(g, 0) + 1
    genres = sorted(genre_counts, key=genre_counts.get, reverse=True)
    counts = [genre_counts[g] for g in genres]

    fig = go.Figure(go.Bar(
        x=counts, y=genres, orientation='h',
        marker=dict(
            color=counts,
            colorscale=[[0, '#2a1f10'], [0.5, '#c94040'], [1, '#e8b84b']],
            showscale=False
        ),
        text=counts, textposition='outside',
        textfont=dict(color='#e8e4de', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e4de', family='DM Sans'),
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        height=420,
    )
    return fig


def rating_histogram(movies_df):
    fig = go.Figure(go.Histogram(
        x=movies_df["imdb_rating"], nbinsx=12,
        marker=dict(color='#e8b84b', opacity=0.8, line=dict(color='#c94040', width=1))
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e4de', family='DM Sans'),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title='IMDb Rating', showgrid=False, color='#6b6b6b'),
        yaxis=dict(title='Count', showgrid=False, color='#6b6b6b'),
        height=220,
    )
    return fig


def radar_chart(metrics):
    categories = ['Precision', 'Recall', 'NDCG', 'Diversity']
    values = [metrics.get('precision', 0), metrics.get('recall', 0),
              metrics.get('ndcg', 0), metrics.get('diversity', 0)]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(232,184,75,0.15)',
        line=dict(color='#e8b84b', width=2),
        marker=dict(color='#e8b84b', size=6)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#6b6b6b'),
                            gridcolor='#2a2a2a', color='#6b6b6b'),
            angularaxis=dict(tickfont=dict(size=11, color='#e8e4de'), gridcolor='#2a2a2a')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e4de', family='DM Sans'),
        margin=dict(l=30, r=30, t=30, b=30),
        height=280,
        showlegend=False
    )
    return fig


def votes_vs_rating_scatter(movies_df):
    fig = px.scatter(
        movies_df, x='imdb_rating', y='votes',
        hover_name='title', size='votes',
        size_max=30, log_y=True,
        color='imdb_rating',
        color_continuous_scale=[[0, '#1a0a0a'], [0.5, '#c94040'], [1, '#e8b84b']],
        labels={'imdb_rating': 'IMDb Rating', 'votes': 'Votes (log)'},
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e4de', family='DM Sans'),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False, color='#6b6b6b'),
        yaxis=dict(showgrid=False, color='#6b6b6b'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='#e8b84b')))
    return fig


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:0.2rem;">
      <span class="hero-title">CineMatch</span>
      <span style="color:#c94040;font-size:1.8rem;">✦</span>
    </div>
    <div class="hero-sub">Hybrid AI · Content-Based + Collaborative Filtering</div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("🎬 Warming up the projector…"):
        recommender, movies_df, ratings_df = load_models()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#e8b84b;margin-bottom:1rem;">
          ⚙️ Settings
        </div>
        """, unsafe_allow_html=True)

        alpha = st.slider(
            "Content ↔ Collaborative Balance",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="0 = pure collaborative, 1 = pure content-based"
        )
        recommender.set_alpha(alpha)

        st.markdown(f"""
        <div style="font-size:0.75rem;color:#6b6b6b;margin-top:-0.5rem;margin-bottom:1rem;">
          Content: {alpha:.0%} &nbsp;·&nbsp; Collaborative: {(1-alpha):.0%}
        </div>
        """, unsafe_allow_html=True)

        top_n = st.slider("Results per query", 4, 12, 8)
        diversity = st.toggle("MMR Diversity Re-ranking", value=True,
                               help="Maximal Marginal Relevance for diverse results")

        st.divider()
        st.markdown('<div style="font-size:0.75rem;color:#6b6b6b;">Dataset</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Movies", len(movies_df))
        col2.metric("Ratings", f"{len(ratings_df):,}")
        col1.metric("Users", ratings_df["user_id"].nunique())
        col2.metric("Sparsity", f"{(1 - len(ratings_df)/(ratings_df['user_id'].nunique()*len(movies_df)))*100:.0f}%")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Find Similar", "🎭 My Watchlist", "📈 Trending", "🔬 Analytics", "🧪 Model Info"
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: Find Similar Movies
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Find Similar Movies</div>', unsafe_allow_html=True)

        movie_options = sorted(movies_df["title"].tolist())
        selected_title = st.selectbox("Choose a movie you love:", movie_options, key="similar_select")

        selected_movie = movies_df[movies_df["title"] == selected_title].iloc[0]

        col_info, col_recs = st.columns([1, 2])

        with col_info:
            st.markdown('<div style="font-size:0.8rem;color:#6b6b6b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Selected Film</div>', unsafe_allow_html=True)
            genres_html = "".join(f'<span class="genre-badge">{g}</span>' for g in selected_movie["genres"].split("|"))
            st.markdown(f"""
            <div class="movie-card" style="border-color:#e8b84b;">
              <div class="movie-title">{selected_movie['title']}</div>
              <div class="movie-meta">🎬 {selected_movie['director']} · {selected_movie['year']}</div>
              <div class="movie-meta">{selected_movie['cast'].replace('|', ', ')}</div>
              <div class="movie-desc">{selected_movie['description']}</div>
              <div>{genres_html}</div>
              <div class="star-row" style="margin-top:0.5rem;">
                <span class="imdb-badge">IMDb</span>
                <span style="color:#f5c518;">{selected_movie['imdb_rating']}/10</span>
                <span style="color:var(--muted);font-size:0.8rem;">{selected_movie['votes']:,} votes</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_recs:
            st.markdown(f'<div style="font-size:0.8rem;color:#6b6b6b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Top {top_n} Recommendations</div>', unsafe_allow_html=True)

            with st.spinner("Computing recommendations…"):
                recs = recommender.recommend_for_movie(
                    selected_movie["movie_id"], top_n=top_n, diversity=diversity
                )

            if recs.empty:
                st.warning("No recommendations found.")
            else:
                for rank, (_, row) in enumerate(recs.iterrows(), 1):
                    render_movie_card(row, show_score=True, rank=rank)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: My Watchlist / Profile Recs
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Personalized Recommendations</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#6b6b6b;font-size:0.85rem;margin-bottom:1rem;">Select movies you\'ve enjoyed — we\'ll find your next favorites.</div>', unsafe_allow_html=True)

        liked_titles = st.multiselect(
            "Movies I've loved:",
            options=sorted(movies_df["title"].tolist()),
            max_selections=10,
            key="liked_movies"
        )

        if liked_titles:
            liked_ids = movies_df[movies_df["title"].isin(liked_titles)]["movie_id"].tolist()

            # Show genre profile
            liked_movies = movies_df[movies_df["movie_id"].isin(liked_ids)]
            genre_freq = {}
            for g_str in liked_movies["genres"]:
                for g in g_str.split("|"):
                    genre_freq[g] = genre_freq.get(g, 0) + 1

            if genre_freq:
                top_genres = sorted(genre_freq, key=genre_freq.get, reverse=True)[:5]
                badges = "".join(f'<span class="genre-badge">{g}</span>' for g in top_genres)
                st.markdown(f'<div style="margin-bottom:1rem;"><span style="color:#6b6b6b;font-size:0.8rem;">Your taste profile: </span>{badges}</div>', unsafe_allow_html=True)

            with st.spinner("Building your profile…"):
                profile_recs = recommender.recommend_for_user_profile(
                    liked_ids, top_n=top_n, diversity=diversity
                )

            if profile_recs.empty:
                st.warning("Add more movies for better recommendations.")
            else:
                st.markdown(f'<div style="font-size:0.8rem;color:#6b6b6b;margin-bottom:0.5rem;">Found {len(profile_recs)} matches for you</div>', unsafe_allow_html=True)
                for rank, (_, row) in enumerate(profile_recs.iterrows(), 1):
                    render_movie_card(row, show_score=True, rank=rank)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#6b6b6b;">
              <div style="font-size:3rem;margin-bottom:1rem;">🎞️</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#3a3a3a;">
                Select movies above to get started
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: Trending / Browse
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Trending & Browse by Genre</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            browse_mode = st.radio("Mode", ["🏆 All-Time Best", "🎭 By Genre"], horizontal=False)

        with col_b:
            if "Genre" in browse_mode:
                all_genres = sorted(set(
                    g for genres in movies_df["genres"] for g in genres.split("|")
                ))
                sel_genres = st.multiselect("Select genres:", all_genres,
                                            default=["Drama"], key="browse_genres")
            else:
                sel_genres = []

        if "Genre" in browse_mode and sel_genres:
            results = recommender.get_genre_recommendations(sel_genres, top_n=top_n)
        else:
            results = recommender.get_trending(top_n=top_n)

        if results.empty:
            st.warning("No movies found for the selected filters.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                render_movie_card(row, show_score=False, rank=rank)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4: Analytics
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Catalog Analytics</div>', unsafe_allow_html=True)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-tile"><div class="metric-val">{len(movies_df)}</div><div class="metric-lbl">Movies</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-tile"><div class="metric-val">{ratings_df["user_id"].nunique()}</div><div class="metric-lbl">Users</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-tile"><div class="metric-val">{len(ratings_df):,}</div><div class="metric-lbl">Ratings</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-tile"><div class="metric-val">{movies_df["imdb_rating"].mean():.1f}</div><div class="metric-lbl">Avg IMDb</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<div style="font-size:0.85rem;color:#6b6b6b;margin-bottom:0.3rem;">Genre Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(genre_distribution_chart(movies_df), use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown('<div style="font-size:0.85rem;color:#6b6b6b;margin-bottom:0.3rem;">Rating Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(rating_histogram(movies_df), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div style="font-size:0.85rem;color:#6b6b6b;margin-bottom:0.3rem;margin-top:0.5rem;">Rating vs Popularity</div>', unsafe_allow_html=True)
            st.plotly_chart(votes_vs_rating_scatter(movies_df), use_container_width=True, config={"displayModeBar": False})

        # Top directors
        st.markdown('<div class="section-header">Top Directors by Avg IMDb</div>', unsafe_allow_html=True)
        dir_stats = movies_df.groupby("director").agg(
            avg_rating=("imdb_rating", "mean"),
            n_films=("movie_id", "count")
        ).reset_index().sort_values("avg_rating", ascending=False).head(10)

        fig_dir = go.Figure(go.Bar(
            x=dir_stats["director"], y=dir_stats["avg_rating"],
            marker=dict(color=dir_stats["avg_rating"],
                        colorscale=[[0,'#2a1f10'],[0.5,'#c94040'],[1,'#e8b84b']]),
            text=[f"{r:.2f}" for r in dir_stats["avg_rating"]],
            textposition="outside", textfont=dict(color="#e8e4de")
        ))
        fig_dir.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8e4de', family='DM Sans'),
            xaxis=dict(showgrid=False, color='#6b6b6b', tickangle=-20),
            yaxis=dict(showgrid=False, range=[7, 9.5], color='#6b6b6b'),
            margin=dict(l=0, r=0, t=20, b=0),
            height=260,
        )
        st.plotly_chart(fig_dir, use_container_width=True, config={"displayModeBar": False})

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5: Model Info
    # ────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Model Architecture & Evaluation</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="movie-card">
              <div style="font-family:'Playfair Display',serif;color:#e8b84b;font-size:1rem;margin-bottom:0.7rem;">
                📄 Content-Based Filtering
              </div>
              <div style="color:#b0a89c;font-size:0.85rem;line-height:1.8;">
                <b style="color:#e8e4de;">Method:</b> TF-IDF Vectorization<br>
                <b style="color:#e8e4de;">Features:</b> Description, Genres (×3), Director (×2), Cast (×2)<br>
                <b style="color:#e8e4de;">N-grams:</b> Unigram + Bigram<br>
                <b style="color:#e8e4de;">Similarity:</b> Cosine similarity<br>
                <b style="color:#e8e4de;">Vocab:</b> 5,000 terms
              </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="movie-card">
              <div style="font-family:'Playfair Display',serif;color:#e8b84b;font-size:1rem;margin-bottom:0.7rem;">
                👥 Collaborative Filtering
              </div>
              <div style="color:#b0a89c;font-size:0.85rem;line-height:1.8;">
                <b style="color:#e8e4de;">Method:</b> Matrix Factorization (Truncated SVD)<br>
                <b style="color:#e8e4de;">Factors:</b> 30 latent dimensions<br>
                <b style="color:#e8e4de;">Matrix:</b> {ratings_df['user_id'].nunique()} × {len(movies_df)} (user–item)<br>
                <b style="color:#e8e4de;">Item sim:</b> Cosine in latent space<br>
                <b style="color:#e8e4de;">Sparsity:</b> {(1 - len(ratings_df)/(ratings_df['user_id'].nunique()*len(movies_df)))*100:.0f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="movie-card" style="margin-top:0.5rem;">
          <div style="font-family:'Playfair Display',serif;color:#e8b84b;font-size:1rem;margin-bottom:0.7rem;">
            🔀 Hybrid Blending
          </div>
          <div style="color:#b0a89c;font-size:0.85rem;line-height:1.8;">
            <b style="color:#e8e4de;">Formula:</b> α × content_score + (1−α) × cf_score + 0.1 × popularity<br>
            <b style="color:#e8e4de;">Alpha (current):</b> {alpha:.2f}<br>
            <b style="color:#e8e4de;">Diversity:</b> Maximal Marginal Relevance (MMR) re-ranking<br>
            <b style="color:#e8e4de;">Cold-start:</b> Falls back to content-based for new users
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Evaluation
        st.markdown('<div class="section-header">Offline Evaluation (Hold-out)</div>', unsafe_allow_html=True)

        eval_button = st.button("▶ Run Evaluation (80 users, K=10)")
        if eval_button or "eval_metrics" in st.session_state:
            if eval_button or "eval_metrics" not in st.session_state:
                with st.spinner("Running evaluation…"):
                    metrics = run_evaluation(recommender, ratings_df)
                    st.session_state["eval_metrics"] = metrics
            else:
                metrics = st.session_state["eval_metrics"]

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(radar_chart(metrics), use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
                for metric, val in metrics.items():
                    color = "#e8b84b" if val > 50 else "#c94040" if val < 20 else "#7a9cc4"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:0.6rem 0;border-bottom:1px solid #1e1e20;">
                      <span style="font-size:0.85rem;text-transform:capitalize;">{metric}@10</span>
                      <span style="font-family:'Playfair Display',serif;font-size:1.2rem;color:{color};">
                        {val:.1f}%
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.72rem;color:#3a3a3a;margin-top:0.8rem;">
                  Train/test split 80/20 per user. Ratings ≥ 4.0 considered relevant.
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
