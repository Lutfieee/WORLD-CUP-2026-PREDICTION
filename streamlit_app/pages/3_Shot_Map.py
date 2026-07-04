"""Shot map analytics page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui import apply_page_config, inject_css, styled_header
from streamlit_app.utils.plotting import pitch_shot_map


apply_page_config()
inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Shot Map Analysis", "Shot location, xG per shot, goal map, density, and zone analysis.")

goals_only = st.toggle("Goals only", value=False)
st.plotly_chart(pitch_shot_map(goals_only=goals_only), use_container_width=True)
st.caption("The visual layer is seeded for portfolio operation; connect event feeds to replace synthetic shots with provider event data.")
