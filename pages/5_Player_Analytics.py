"""Player analytics page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.components.ui import inject_css, styled_header
from streamlit_app.utils.data_loader import load_table


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Player Analytics", "Player rating, goals, assists, xG, xA, minutes, touches, and defensive action.")

players = load_table("players")
team = st.selectbox("Team", sorted(players["team"].unique()))
filtered = players[players["team"] == team].sort_values("rating", ascending=False)
st.dataframe(filtered, use_container_width=True, hide_index=True)
fig = px.scatter(filtered, x="xg", y="xa", size="minutes", color="position", hover_name="player", template="plotly_dark", title="Player xG and xA Profile")
st.plotly_chart(fig, use_container_width=True)
