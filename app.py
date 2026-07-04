"""Main Streamlit entrypoint for the AI World Cup 2026 predictor."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from scraping.live_worldcup import fetch_live_games, fetch_live_teams, live_summary
from streamlit_app.components.ui import apply_page_config, format_probability_columns, inject_css, kpi_card, plotly_template, styled_header
from streamlit_app.utils.data_loader import data_ready, load_table


def ensure_data() -> None:
    """Run the pipeline from Streamlit when processed data is missing."""

    if data_ready():
        return
    with st.spinner("Preparing analytics pipeline, models, and predictions..."):
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)


def main() -> None:
    """Render the executive dashboard."""

    apply_page_config()
    theme = st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True)
    inject_css(theme)
    template = plotly_template(theme)
    ensure_data()
    predictions = load_table("predictions")
    matches = load_table("matches")
    simulation = load_table("simulation")
    teams = load_table("teams")
    live_games = fetch_live_games()
    live_teams = fetch_live_teams()
    summary = live_summary(live_games)
    source_label = "Live World Cup feed" if summary["available"] else "Local ML pipeline"
    total_goals = summary["total_goals"] if summary["available"] else int(matches["home_goals"].sum() + matches["away_goals"].sum())
    remaining_matches = summary["remaining_matches"] if summary["available"] else len(predictions)
    last_update = summary["last_update"] if summary["available"] else pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    styled_header("FIFA World Cup 2026 Predictor by lutfie", "Professional football analytics dashboard for match, score, and tournament forecasting.")
    st.caption(f"Data source: {source_label}. FIFA ranking baseline: official 11 June 2026 release; next official update is 20 July 2026.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Remaining Matches", str(remaining_matches), "Live schedule if API is reachable")
    with c2:
        kpi_card("Total Goals", str(total_goals), "Current tournament finished matches")
    with c3:
        kpi_card("Average xG", f"{(matches['home_xg'].mean() + matches['away_xg'].mean()) / 2:.2f}", "Per team per match")
    with c4:
        favorite = simulation.iloc[0]["team"] if not simulation.empty else "-"
        kpi_card("Champion Favorite", str(favorite), "Monte Carlo leader")
    with c5:
        kpi_card("Last Update", str(last_update).replace(" UTC", ""), "UTC refresh timestamp")

    st.markdown("<div class='section-title'>Remaining Match Predictions</div>", unsafe_allow_html=True)
    display_cols = [
        "date",
        "home_team",
        "away_team",
        "stage",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "expected_home_goals",
        "expected_away_goals",
        "predicted_score",
        "scoreline_probability",
        "confidence",
    ]
    st.dataframe(format_probability_columns(predictions[display_cols]), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Champion Probability</div>", unsafe_allow_html=True)
    fig = px.bar(simulation.head(12), x="champion", y="team", orientation="h", color="champion", color_continuous_scale="Viridis", template=template)
    fig.update_layout(height=520, xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)

    if not live_teams.empty:
        with st.expander("Qualified Teams From Live Feed", expanded=False):
            st.dataframe(live_teams.sort_values(["group", "team"]), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Team Power Index</div>", unsafe_allow_html=True)
    power = teams.assign(power_index=lambda df: df["elo"] / 22 + (101 - df["fifa_rank"]) * 0.55 + df["attack"] * 18 - df["defense"] * 10)
    fig = px.scatter(power, x="elo", y="fifa_rank", size="attack", color="confederation", hover_name="team", template=template, title="Elo, FIFA Ranking, and Attack Profile")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
