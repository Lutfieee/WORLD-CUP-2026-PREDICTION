"""Tournament simulation page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from streamlit_app.components.ui import format_probability_columns, inject_css, styled_header
from streamlit_app.utils.data_loader import load_table


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Tournament Simulation", "Monte Carlo probabilities for champion, runner-up, semi-final, and quarter-final reach.")

simulation = load_table("simulation")
stage = st.selectbox("Probability", ["champion", "runner_up", "semi_final", "quarter_final"])
fig = px.bar(simulation.sort_values(stage, ascending=False).head(12), x=stage, y="team", orientation="h", color=stage, color_continuous_scale="Viridis", template="plotly_dark")
fig.update_layout(xaxis_tickformat=".0%", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(format_probability_columns(simulation), use_container_width=True, hide_index=True)
