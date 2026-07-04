"""Team comparison page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui import inject_css, styled_header
from streamlit_app.utils.data_loader import load_table
from streamlit_app.utils.plotting import radar_chart


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Team Comparison", "Radar comparison for goal model inputs, ranking, Elo, possession, and passing.")

teams = load_table("teams")
names = sorted(teams["team"].unique())
t1 = st.selectbox("Team A", names, index=0)
t2 = st.selectbox("Team B", names, index=1)
selected = teams[teams["team"].isin([t1, t2])]
metrics = ["attack", "defense", "possession", "pass_accuracy", "elo", "fifa_rank"]
st.plotly_chart(radar_chart(selected, "team", metrics, "Team Radar"), use_container_width=True)
st.dataframe(selected, use_container_width=True, hide_index=True)
