import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Yelp Hybrid Recommender",
    page_icon="⭐",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# =========================================================
# CARGA DE DATOS
# =========================================================

@st.cache_data
def load_data():

    item_features = pd.read_csv(BASE_DIR / "item_features.csv")
    context_stats = pd.read_csv(BASE_DIR / "context_stats.csv")

    metricas_regresion = pd.read_csv(BASE_DIR / "metricas_regresion.csv")
    metricas_ranking = pd.read_csv(BASE_DIR / "metricas_ranking.csv")

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
# ARTEFACTOS
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
# FUNCIONES MODELO
# =========================================================

def baseline_prediction(user_id, business_id):

    if user_id in user_mean and business_id in business_mean:
        return (user_mean[user_id] + business_mean[business_id]) / 2

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
        pred = baseline_prediction(user_id, business_id)

    return float(np.clip(pred, 1, 5))


def build_explanation(row):

    reasons = []

    if row["pred_svd"] >= 4:
        reasons.append(
            "usuarios con preferencias similares calificaron este negocio positivamente"
        )

    if row["pred_context"] >= 4:
        reasons.append(
            "el contexto seleccionado presenta alta afinidad con este tipo de negocio"
        )

    if row["review_count"] >= 100:
        reasons.append(
            "el negocio posee una alta cantidad de reseñas"
        )

    if row["business_avg_stars"] >= 4:
        reasons.append(
            "el negocio tiene una excelente calificación promedio"
        )

    if not reasons:
        reasons.append(
            "el sistema encontró afinidad moderada entre el usuario y el negocio"
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

    # SVD
    candidates["pred_svd"] = [
        predict_svd(user_id, b)
        for b in candidates["business_id"]
    ]

    # CONTEXTO
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
        + (1 - best_alpha) * candidates["pred_context"],
        1,
        5
    )

    # EXPLICACIÓN
    candidates["explicacion"] = candidates.apply(
        build_explanation,
        axis=1
    )

    return (
        candidates
        .sort_values("score_hybrid", ascending=False)
        .head(top_n)
    )

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Configuración")

user_options = list(user_to_idx.keys())[:500]

user_id = st.sidebar.selectbox(
    "Usuario",
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
    "Ciudad",
    cities
)

category = st.sidebar.selectbox(
    "Categoría",
    categories
)

is_weekend = st.sidebar.checkbox(
    "Fin de semana"
)

top_n = st.sidebar.slider(
    "Número de recomendaciones",
    5,
    20,
    10
)

# =========================================================
# HEADER
# =========================================================

st.title("⭐ Sistema Híbrido de Recomendación Yelp")

st.markdown("""
Este sistema combina:

- **Filtrado colaborativo por factorización matricial (SVD)**
- **Recomendación sensible al contexto**
- **Modelo híbrido ponderado**

El objetivo es generar recomendaciones personalizadas
de negocios considerando preferencias históricas y contexto.
""")

# =========================================================
# MÉTRICAS
# =========================================================

st.subheader("📊 Evaluación del modelo")

col1, col2, col3, col4 = st.columns(4)

rmse = metricas_regresion["RMSE"].iloc[0]
mae = metricas_regresion["MAE"].iloc[0]

precision = metricas_ranking["precision@10"].iloc[0]
ndcg = metricas_ranking["ndcg@10"].iloc[0]

col1.metric("RMSE", f"{rmse:.3f}")
col2.metric("MAE", f"{mae:.3f}")
col3.metric("precision@10", f"{precision:.3f}")
col4.metric("ndcg@10", f"{ndcg:.3f}")

# =========================================================
# GENERAR RECOMENDACIONES
# =========================================================

if st.button("🚀 Generar recomendaciones"):

    recs = recommend(
        user_id,
        city,
        category,
        int(is_weekend),
        top_n
    )

    if recs.empty:

        st.warning(
            "No se encontraron recomendaciones."
        )

    else:

        st.subheader("🎯 Recomendaciones personalizadas")

        for _, row in recs.iterrows():

            with st.container():

                st.markdown("---")

                col1, col2 = st.columns([3, 1])

                with col1:

                    st.markdown(
                        f"## {row['name']}"
                    )

                    st.write(
                        f"📍 {row['city']}, {row['state']}"
                    )

                    st.write(
                        f"🍽️ Categoría: {row['main_category']}"
                    )

                    st.write(
                        f"⭐ Rating promedio: "
                        f"{row['business_avg_stars']:.1f}"
                    )

                    st.write(
                        f"📝 Reviews: "
                        f"{int(row['review_count'])}"
                    )

                with col2:

                    st.metric(
                        "Score híbrido",
                        f"{row['score_hybrid']:.2f}"
                    )

                st.info(
                    f"""
                    **Explicación de la recomendación**

                    {row['explicacion']}
                    """
                )

                with st.expander(
                    "Ver detalles técnicos"
                ):

                    st.write(
                        f"SVD Score: "
                        f"{row['pred_svd']:.2f}"
                    )

                    st.write(
                        f"Context Score: "
                        f"{row['pred_context']:.2f}"
                    )

                    st.write(
                        f"Alpha híbrido: "
                        f"{best_alpha:.2f}"
                    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Proyecto académico — Sistemas de Recomendación

Modelo híbrido:
SVD + Context-Aware Recommendation
""")
