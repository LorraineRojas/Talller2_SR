import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    item_features = pd.read_csv(BASE_DIR / "item_features.csv")
    context_stats = pd.read_csv(BASE_DIR / "context_stats.csv")
    with open(BASE_DIR / "svd_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return item_features, context_stats, artifacts

item_features, context_stats, artifacts = load_data()

global_mean = artifacts["global_mean"]
user_mean = artifacts["user_mean"]
business_mean = artifacts["business_mean"]
user_to_idx = artifacts["user_to_idx"]
item_to_idx = artifacts["item_to_idx"]
user_factors = artifacts["user_factors"]
item_factors = artifacts["item_factors"]
best_alpha = artifacts["best_alpha"]


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
        pred = float(np.dot(user_factors[u_idx], item_factors[i_idx]))
    else:
        pred = baseline_prediction(user_id, business_id)
    return float(np.clip(pred, 1, 5))


def recommend(user_id, city=None, main_category=None, is_weekend=0, top_n=10):
    candidates = item_features.copy()
    if city and city != "Todas":
        candidates = candidates[candidates["city"] == city].copy()
    if main_category and main_category != "Todas":
        candidates = candidates[candidates["main_category"] == main_category].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["is_weekend"] = int(is_weekend)
    candidates["pred_svd"] = [predict_svd(user_id, b) for b in candidates["business_id"]]
    candidates = candidates.merge(
        context_stats[["city", "main_category", "is_weekend", "context_score"]],
        on=["city", "main_category", "is_weekend"], how="left"
    )
    candidates["pred_context"] = candidates["context_score"].fillna(global_mean)
    candidates["score_hybrid"] = np.clip(best_alpha * candidates["pred_svd"] + (1 - best_alpha) * candidates["pred_context"], 1, 5)
    candidates["explicacion"] = candidates.apply(
        lambda r: f"SVD={r['pred_svd']:.2f}, contexto={r['pred_context']:.2f}, score final={r['score_hybrid']:.2f}", axis=1
    )
    return candidates.sort_values("score_hybrid", ascending=False).head(top_n)

st.title("Sistema de Recomendación Híbrido Yelp")
st.write("Modelo híbrido: SVD por factorización matricial + recomendación sensible al contexto.")
user_options = list(user_to_idx.keys())[:500]
user_id = st.selectbox("Usuario", user_options)
cities = ["Todas"] + sorted(item_features["city"].dropna().unique().tolist())
categories = ["Todas"] + sorted(item_features["main_category"].dropna().unique().tolist())
city = st.selectbox("Ciudad", cities)
category = st.selectbox("Categoría", categories)
is_weekend = st.checkbox("Contexto: fin de semana")
top_n = st.slider("Número de recomendaciones", 5, 20, 10)
if st.button("Generar recomendaciones"):
    recs = recommend(user_id, city, category, int(is_weekend), top_n)
    if recs.empty:
        st.warning("No hay recomendaciones para esos filtros.")
    else:
        st.dataframe(recs[["name", "city", "state", "main_category", "business_avg_stars", "review_count", "score_hybrid", "explicacion"]])
