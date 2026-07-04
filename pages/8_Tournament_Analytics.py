"""Tournament analytics page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.components.ui import inject_css, styled_header
from streamlit_app.utils.data_loader import load_table


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Tournament Analytics", "Goal distribution, xG distribution, possession ranking, passing ranking, top scorer, and clean sheet view.")

matches = load_table("matches")
players = load_table("players")
fig = px.histogram(matches, x="home_goals", nbins=8, template="plotly_dark", title="Goal Distribution")
st.plotly_chart(fig, use_container_width=True)
c1, c2 = st.columns(2)
with c1:
    st.dataframe(players.sort_values("goals", ascending=False).head(10), use_container_width=True, hide_index=True)
with c2:
    st.dataframe(players.sort_values("assists", ascending=False).head(10), use_container_width=True, hide_index=True)
