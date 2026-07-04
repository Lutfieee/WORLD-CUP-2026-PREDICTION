"""xG analytics page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.components.ui import inject_css, styled_header
from streamlit_app.utils.data_loader import load_table
from streamlit_app.utils.plotting import xg_timeline


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("xG Analysis", "Cumulative xG, finishing efficiency, and goal versus expected goal trends.")

matches = load_table("matches")
st.plotly_chart(xg_timeline(), use_container_width=True)
team_xg = matches.groupby("home_team", as_index=False).agg(goals=("home_goals", "mean"), xg=("home_xg", "mean"))
fig = px.scatter(team_xg, x="xg", y="goals", hover_name="home_team", trendline="ols", template="plotly_dark", title="Goal vs xG Efficiency")
st.plotly_chart(fig, use_container_width=True)
