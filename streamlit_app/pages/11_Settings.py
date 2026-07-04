"""Settings and data refresh page."""

from __future__ import annotations

import subprocess
import sys

import streamlit as st

from streamlit_app.components.ui import apply_page_config, inject_css, styled_header
from streamlit_app.utils.data_loader import data_ready


apply_page_config()
inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Settings", "Refresh data, retrain models, and inspect operational readiness.")

st.write("Pipeline status:", "Ready" if data_ready() else "Not generated")
simulations = st.select_slider("Monte Carlo simulation scale", options=[10000, 50000, 100000], value=10000)
st.caption(f"Selected scale: {simulations:,}. Update DEFAULT_SIMULATIONS in `.env` for scheduled production runs.")
if st.button("Refresh Pipeline"):
    with st.spinner("Running ETL, training, prediction, and simulation..."):
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)
    st.success("Pipeline refreshed.")
