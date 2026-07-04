"""Interactive match predictor page."""

from __future__ import annotations

import streamlit as st

from models.explain import local_explanation
from streamlit_app.components.ui import apply_page_config, inject_css, kpi_card, styled_header
from streamlit_app.utils.data_loader import load_artifact, load_table
from streamlit_app.utils.plotting import probability_bar


apply_page_config()
inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Match Predictor", "Select a scheduled fixture and inspect probabilities, expected goals, scoreline, confidence, and model rationale.")

predictions = load_table("predictions")
features = load_table("fixture_features")
classifier = load_artifact("saved_models/match_outcome_classifier.joblib")

fixture_label = st.selectbox("Fixture", [f"{r.home_team} vs {r.away_team}" for r in predictions.itertuples()])
idx = [f"{r.home_team} vs {r.away_team}" for r in predictions.itertuples()].index(fixture_label)
row = predictions.iloc[idx]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card(row["home_team"], f"{row.get('prob_home_win', 0):.0%}", "Home win probability")
with c2:
    kpi_card("Draw", f"{row.get('prob_draw', 0):.0%}", "Regulation draw probability")
with c3:
    kpi_card(row["away_team"], f"{row.get('prob_away_win', 0):.0%}", "Away win probability")
with c4:
    kpi_card("Predicted Score", str(row["predicted_score"]), f"Confidence {row['confidence']:.0%}")

left, right = st.columns([1.1, 1])
with left:
    st.plotly_chart(probability_bar(row), use_container_width=True)
with right:
    st.metric("Expected Goals", f"{row['expected_home_goals']:.2f} - {row['expected_away_goals']:.2f}")
    st.progress(float(row["confidence"]))
    st.caption("Confidence is the highest class probability returned by the outcome classifier.")

st.markdown("#### Local Explanation")
if classifier is not None and not features.empty:
    st.dataframe(local_explanation(classifier, features.iloc[idx]), use_container_width=True, hide_index=True)
else:
    st.info("Run `python run_pipeline.py` to generate model artifacts.")
