"""Match preview page with team radar and head-to-head context."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_app.components.ui import inject_css, kpi_card, styled_header
from streamlit_app.utils.data_loader import load_table
from streamlit_app.utils.plotting import radar_chart


inject_css(st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True))
styled_header("Match Preview", "Team profiles, rankings, form, and tactical radar before kickoff.")

teams = load_table("teams")
matches = load_table("matches")
team_names = sorted(teams["team"].unique())
home, away = st.columns(2)
with home:
    team_a = st.selectbox("Team A", team_names, index=team_names.index("Spain") if "Spain" in team_names else 0)
with away:
    team_b = st.selectbox("Team B", team_names, index=team_names.index("Portugal") if "Portugal" in team_names else 1)

selected = teams[teams["team"].isin([team_a, team_b])].copy()
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("FIFA Ranking", f"{int(selected.iloc[0]['fifa_rank'])} / {int(selected.iloc[1]['fifa_rank'])}", "Lower is better")
with c2:
    kpi_card("Elo Rating", f"{int(selected.iloc[0]['elo'])} / {int(selected.iloc[1]['elo'])}", "Long-term team strength")
with c3:
    kpi_card("Pass Accuracy", f"{selected.iloc[0]['pass_accuracy']:.1f}% / {selected.iloc[1]['pass_accuracy']:.1f}%", "Possession reliability")
with c4:
    kpi_card("Experience", f"{int(selected.iloc[0]['tournament_experience'])} / {int(selected.iloc[1]['tournament_experience'])}", "World Cup appearances")

metrics = ["attack", "defense", "possession", "pass_accuracy", "elo", "tournament_experience"]
st.plotly_chart(radar_chart(selected, "team", metrics, "Team Radar"), use_container_width=True)

h2h = matches[((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) | ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))]
st.markdown("#### Head to Head")
st.dataframe(h2h.tail(8) if not h2h.empty else pd.DataFrame({"message": ["No direct historical seed matches found."]}), use_container_width=True, hide_index=True)
