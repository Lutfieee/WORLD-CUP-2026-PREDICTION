"""Player comparison page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui import apply_page_config, inject_css, styled_header
from streamlit_app.utils.data_loader import load_table
from streamlit_app.utils.plotting import radar_chart


apply_page_config()
inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Player Comparison", "Compare two players across attacking, passing, defensive, and rating dimensions.")

players = load_table("players")
names = sorted(players["player"].unique())
p1 = st.selectbox("Player A", names, index=0)
p2 = st.selectbox("Player B", names, index=min(12, len(names) - 1))
selected = players[players["player"].isin([p1, p2])]
metrics = ["goals", "assists", "xg", "xa", "pass_accuracy", "tackles", "interceptions", "rating"]
st.plotly_chart(radar_chart(selected, "player", metrics, "Player Radar"), use_container_width=True)
st.dataframe(selected, use_container_width=True, hide_index=True)
