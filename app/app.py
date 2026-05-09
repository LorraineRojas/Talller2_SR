import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Yelp Smart Recommender",
    page_icon="🍴",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# =========================================================
# ESTILOS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-size: 18px;
    font-weight: 600;
}

.card {
    background-color: #1e1e1e;
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    border: 1px solid #333333;
}

.card-title {
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.card-subtitle {
    color: #bbbbbb;
    margin-bottom: 1rem;
}

.score-box {
    background-color: #262730;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}

.explanation-box {
    background-color: #202020;
    padding: 1rem;
    border-radius: 12px;
    border-left: 5px solid #ff4b4b;
    margin-top: 1rem;
}

.small-text {
    color: #bbbbbb;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CARGA DE DATOS
# =========================================================

@st.cache_data
def load_data():

    item_features = pd.read_csv(BASE_DIR / "item_features.csv")
    context_stats = pd.read_csv(BASE_DIR / "context_stats.csv")

    metricas_regresion = pd.read_csv(
        BASE_DIR / "metricas_regresion.csv"
    )

    metricas_ranking = pd.read_csv(
        BASE_DIR / "metricas_ranking.csv"
    )

    with open(BASE_DIR / "svd_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    return (
        item_features,
        context_stats,
        metricas_regresion,
        metricas_ranking,
        artifacts
    )


(
    item_features,
    context_stats,
    metricas_regresion,
    metricas_ranking,
    artifacts
) = load_data()

# =========================================================
# ARTEFACTOS DEL MODELO
# =========================================================

global_mean = artifacts["global_mean"]

user_mean = artifacts["user_mean"]
business_mean = artifacts["business_mean"]

user_to_idx = artifacts["user_to_idx"]
item_to_idx = artifacts["item_to_idx"]

user_factors = artifacts["user_factors"]
item_factors = artifacts["item_factors"]

best_alpha = artifacts["best_alpha"]

# =========================================================
# FUNCIONES DEL MODELO
# =========================================================

def baseline_prediction(user_id, business_id):

    if user_id in user_mean and business_id in business_mean:
        return (
            user_mean[user_id]
            + business_mean[business_id]
        ) / 2

    if user_id in user_mean:
        return user_mean[user_id]

    if business_id in business_mean:
        return business_mean[business_id]

    return global_mean


def predict_svd(user_id, business_id):

    if user_id in user_to_idx and business_id in item_to_idx:

        u_idx = user_to_idx[user_id]
        i_idx = item_to_idx[business_id]

        pred = float(
            np.dot(
                user_factors[u_idx],
                item_factors[i_idx]
            )
        )

    else:
        pred = baseline_prediction(
            user_id,
            business_id
        )

    return float(np.clip(pred, 1, 5))


def build_explanation(row):

    reasons = []

    if row["pred_svd"] >= 4:
        reasons.append(
            "usuarios con gustos similares disfrutaron este lugar"
        )

    if row["pred_context"] >= 4:
        reasons.append(
            "este tipo de negocio es popular en el contexto seleccionado"
        )

    if row["business_avg_stars"] >= 4:
        reasons.append(
            "el negocio tiene excelentes calificaciones"
        )

    if row["review_count"] >= 100:
        reasons.append(
            "muchas personas han visitado este lugar"
        )

    if not reasons:
        reasons.append(
            "el sistema encontró afinidad entre tus preferencias y este negocio"
        )

    return " • ".join(reasons)


def recommend(
    user_id,
    city=None,
    main_category=None,
    is_weekend=0,
    top_n=10
):

    candidates = item_features.copy()

    # FILTROS
    if city and city != "Todas":
        candidates = candidates[
            candidates["city"] == city
        ].copy()

    if main_category and main_category != "Todas":
        candidates = candidates[
            candidates["main_category"] == main_category
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates["is_weekend"] = int(is_weekend)

    # SCORE SVD
    candidates["pred_svd"] = [
        predict_svd(user_id, b)
        for b in candidates["business_id"]
    ]

    # SCORE CONTEXTUAL
    candidates = candidates.merge(
        context_stats[
            [
                "city",
                "main_category",
                "is_weekend",
                "context_score"
            ]
        ],
        on=[
            "city",
            "main_category",
            "is_weekend"
        ],
        how="left"
    )

    candidates["pred_context"] = (
        candidates["context_score"]
        .fillna(global_mean)
    )

    # SCORE HÍBRIDO
    candidates["score_hybrid"] = np.clip(
        best_alpha * candidates["pred_svd"]
        + (1 - best_alpha)
        * candidates["pred_context"],
        1,
        5
    )

    # EXPLICACIONES
    candidates["explicacion"] = candidates.apply(
        build_explanation,
        axis=1
    )

    return (
        candidates
        .sort_values(
            "score_hybrid",
            ascending=False
        )
        .head(top_n)
    )

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Personaliza tu experiencia")

user_options = list(user_to_idx.keys())[:500]

user_id = st.sidebar.selectbox(
    "👤 Perfil de usuario",
    user_options
)

cities = ["Todas"] + sorted(
    item_features["city"]
    .dropna()
    .unique()
    .tolist()
)

categories = ["Todas"] + sorted(
    item_features["main_category"]
    .dropna()
    .unique()
    .tolist()
)

city = st.sidebar.selectbox(
    "📍 Ciudad",
    cities
)

category = st.sidebar.selectbox(
    "🍽️ Tipo de negocio",
    categories
)

is_weekend = st.sidebar.checkbox(
    "🌙 Estoy buscando planes para fin de semana"
)

top_n = st.sidebar.slider(
    "⭐ Número de recomendaciones",
    5,
    20,
    10
)

# =========================================================
# HEADER PRINCIPAL
# =========================================================

st.title("🍴 Encuentra tu próximo lugar favorito")

st.markdown("""
Descubre restaurantes y negocios recomendados especialmente para ti,
considerando tus preferencias y el contexto de tu visita.
""")

# =========================================================
# BOTÓN PRINCIPAL
# =========================================================

if st.button("🔍 Descubrir recomendaciones"):

    recs = recommend(
        user_id,
        city,
        category,
        int(is_weekend),
        top_n
    )

    if recs.empty():

        st.warning(
            "No encontramos recomendaciones con esos filtros."
        )

    else:

        st.subheader("✨ Recomendaciones personalizadas")

        for _, row in recs.iterrows():

            stars = "⭐" * int(
                round(row["business_avg_stars"])
            )

            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        {row['name']}
                    </div>

                    <div class="card-subtitle">
                        📍 {row['city']}, {row['state']}
                    </div>

                    <div>
                        🍽️ <b>Categoría:</b> {row['main_category']}
                    </div>

                    <div style="margin-top:0.5rem;">
                        {stars}
                        ({row['business_avg_stars']:.1f})
                    </div>

                    <div style="margin-top:0.5rem;">
                        📝 {int(row['review_count'])} reseñas
                    </div>

                    <div class="explanation-box">
                        <b>✨ ¿Por qué te lo recomendamos?</b><br><br>
                        {row['explicacion']}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(
                "Ver detalles de la recomendación"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Score híbrido",
                        f"{row['score_hybrid']:.2f}"
                    )

                with col2:
                    st.metric(
                        "Afinidad colaborativa",
                        f"{row['pred_svd']:.2f}"
                    )

                with col3:
                    st.metric(
                        "Afinidad contextual",
                        f"{row['pred_context']:.2f}"
                    )

# =========================================================
# INFORMACIÓN TÉCNICA
# =========================================================

with st.expander("ℹ️ Información técnica del sistema"):

    st.markdown("""
    Este sistema utiliza un modelo híbrido de recomendación que combina:

    - Filtrado colaborativo mediante factorización matricial (SVD)
    - Recomendación sensible al contexto
    - Fusión híbrida ponderada
    """)

    rmse = metricas_regresion["rmse"].iloc[0]
    mae = metricas_regresion["mae"].iloc[0]

    precision = metricas_ranking["precision@10"].iloc[0]
    recall = metricas_ranking["recall@10"].iloc[0]
    ndcg = metricas_ranking["ndcg@10"].iloc[0]

    st.subheader("📊 Métricas del modelo")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("RMSE", f"{rmse:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("Precision@10", f"{precision:.3f}")
    c4.metric("Recall@10", f"{recall:.3f}")
    c5.metric("NDCG@10", f"{ndcg:.3f}")

    st.markdown(
        f"""
        ### Configuración híbrida

        El sistema utiliza un parámetro de combinación:

        α = {best_alpha:.2f}

        donde:

        - α controla el peso del componente colaborativo
        - (1-α) controla el peso contextual
        """
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Proyecto académico — Sistemas de Recomendación

Modelo híbrido para recomendación personalizada de negocios Yelp.
""")
