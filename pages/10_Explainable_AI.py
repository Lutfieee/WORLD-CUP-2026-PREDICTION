"""Explainable AI dashboard page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from models.explain import feature_importance_frame
from streamlit_app.components.ui import inject_css, styled_header
from streamlit_app.utils.data_loader import load_artifact, load_table


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Explainable AI", "Global feature importance and local model explanation for match predictions.")

classifier = load_artifact("saved_models/match_outcome_classifier.joblib")
features = load_table("features")
if classifier is None or features.empty:
    st.info("Run `python run_pipeline.py` first.")
else:
    importance = feature_importance_frame(classifier, features)
    fig = px.bar(importance, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis", template="plotly_dark", title="SHAP-style Global Importance")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance, use_container_width=True, hide_index=True)
    st.caption("Install `shap` and extend `models/explain.py` for exact SHAP waterfall objects on tree models.")
