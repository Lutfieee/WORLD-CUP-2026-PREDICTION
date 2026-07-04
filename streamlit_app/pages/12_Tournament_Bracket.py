"""Tournament bracket visualization and simulation."""

from __future__ import annotations

import streamlit as st

from config.settings import settings
from feature_engineering.features import build_feature_table, build_fixture_features
from models.bracket import advance_tournament_round, predict_full_bracket
from models.predict import predict_fixtures
from models.simulate import simulate_knockout
from preprocessing.cleaning import clean_matches, store_processed_tables
from streamlit_app.components.ui import apply_page_config, inject_css, styled_header
from streamlit_app.utils.data_loader import load_artifact, load_table


apply_page_config()
theme = st.sidebar.radio("Theme", ["Dark", "Light", "Auto"], index=0, horizontal=True)
inject_css(theme)

styled_header("🏆 Tournament Bracket", "Full knockout tree from Round of 16 to the Champion — simulated in real-time using ML predictions.")

# ── Load current state ──────────────────────────────────────────────────────────
fixtures   = load_table("fixtures")
matches    = load_table("matches")
teams      = load_table("teams")
players    = load_table("players")
classifier = load_artifact("saved_models/match_outcome_classifier.joblib")
regressors = load_artifact("saved_models/score_regressors.joblib")

dark        = theme in {"Dark", "Auto"}
bg_card     = "#111c27" if dark else "#ffffff"
bg_finished = "#0d1c14" if dark else "#edfbf4"
border_act  = "rgba(53,208,127,.40)"
border_fin  = "rgba(53,208,127,.22)"
text        = "#f6fbff" if dark else "#182230"
muted       = "#9aa4b2"
accent      = "#35d07f"

# ── Top control bar ─────────────────────────────────────────────────────────────
if not fixtures.empty:
    current_stage = fixtures.iloc[0].get("stage", "Unknown")
    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.markdown(
            f"<div style='font-size:18px; font-weight:700; color:{accent}; margin-bottom:4px;'>"
            f"Current Round: {current_stage}</div>"
            f"<div style='font-size:13px; color:{muted};'>Press the button to simulate results and generate the next round fixtures.</div>",
            unsafe_allow_html=True,
        )
    with c_right:
        if st.button("⚡ Advance to Next Round", use_container_width=True):
            with st.spinner(f"Simulating {current_stage}..."):
                new_matches, new_fixtures = advance_tournament_round(
                    fixtures, matches, teams, players, classifier, regressors
                )
                new_matches_clean = clean_matches(new_matches)
                feature_frame = build_feature_table(new_matches_clean, teams, players)
                fixture_features = build_fixture_features(new_fixtures, new_matches_clean, teams, players) if not new_fixtures.empty else fixture_features
                if not new_fixtures.empty:
                    predictions = predict_fixtures(fixture_features, classifier, regressors)
                    simulation  = simulate_knockout(predictions, simulations=settings.default_simulations, seed=settings.random_state)
                else:
                    import pandas as pd
                    predictions = pd.DataFrame()
                    simulation  = pd.DataFrame()
                store_processed_tables({
                    "matches":          new_matches_clean,
                    "fixtures":         new_fixtures,
                    "teams":            teams,
                    "players":          players,
                    "features":         feature_frame,
                    "fixture_features": fixture_features if not new_fixtures.empty else pd.DataFrame(),
                    "predictions":      predictions,
                    "simulation":       simulation,
                })
            st.cache_data.clear()
            st.rerun()
elif matches[matches["stage"].isin(["Round of 16", "Quarter-final", "Semi-final", "Final"])].empty:
    st.info("Run the pipeline first to generate tournament fixtures.")
else:
    st.success("🏆 The tournament has concluded! See the full bracket below.")

st.divider()

# ── Predict full bracket ─────────────────────────────────────────────────────────
if not fixtures.empty and classifier and regressors:
    bracket = predict_full_bracket(fixtures, matches, teams, players, classifier, regressors)
else:
    bracket = {}

# ── Build history from completed knock-out stages in matches ────────────────────
knockout_stages = ["Round of 16", "Quarter-final", "Semi-final", "Final"]
if "stage" in matches.columns and "status" in matches.columns:
    done_matches = matches[(matches["status"] == "FINISHED") & (matches["stage"].isin(knockout_stages))].copy()
else:
    done_matches = matches.iloc[0:0]

history: dict[str, list[dict]] = {}
for _, row in done_matches.iterrows():
    stage = str(row.get("stage", ""))
    home, away = str(row["home_team"]), str(row["away_team"])
    hg, ag = int(row.get("home_goals", 0)), int(row.get("away_goals", 0))
    winner = home if hg > ag else away
    history.setdefault(stage, []).append({
        "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag, "winner": winner,
    })

# ── Render HTML bracket ──────────────────────────────────────────────────────────
def render_matchup(home, away, s_home, s_away, winner, finished=False):
    h_w = winner == home
    a_w = winner == away
    border = border_fin if finished else border_act
    bg = bg_finished if finished else bg_card
    badge = f"<span style='font-size:10px;background:rgba(53,208,127,.15);color:{accent};border-radius:4px;padding:1px 5px;margin-left:6px;'>✓</span>"
    return f"""
    <div style="background:{bg};border:1px solid {border};border-radius:10px;padding:10px 12px;
                margin:8px 0;color:{text};box-shadow:0 4px 18px rgba(0,0,0,.14);">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;
                  border-bottom:1px solid rgba(154,164,178,.12);{'font-weight:700;color:'+accent if h_w else ''}">
        <span>{home}{badge if h_w and finished else ''}</span>
        <span style="font-size:18px;font-weight:700;">{s_home}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;
                  {'font-weight:700;color:'+accent if a_w else ''}">
        <span>{away}{badge if a_w and finished else ''}</span>
        <span style="font-size:18px;font-weight:700;">{s_away}</span>
      </div>
    </div>"""

html_parts = ["<div style='display:flex;flex-direction:row;width:100%;overflow-x:auto;gap:4px;padding:8px 0;'>"]

for stage in knockout_stages:
    html_parts.append(
        f"<div style='display:flex;flex-direction:column;justify-content:space-around;"
        f"flex:1;min-width:210px;padding:0 8px;'>"
        f"<div style='text-align:center;color:{muted};font-size:12px;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:12px;font-weight:600;'>{stage}</div>"
    )

    # Finished matches (history)
    if stage in history:
        for m in history[stage]:
            html_parts.append(render_matchup(
                m["home_team"], m["away_team"],
                str(m["home_goals"]), str(m["away_goals"]),
                m["winner"], finished=True,
            ))

    # Predicted matches (bracket)
    elif stage in bracket:
        for m in bracket[stage]:
            sc = str(m.get("predicted_score", ""))
            s_h = sc.split("-")[0] if "-" in sc else "?"
            s_a = sc.split("-")[1] if "-" in sc else "?"
            html_parts.append(render_matchup(
                m["home_team"], m["away_team"], s_h, s_a, m["winner"], finished=False,
            ))
    else:
        html_parts.append(
            f"<div style='color:{muted};font-size:13px;text-align:center;padding:20px 0;'>"
            f"Awaiting previous round…</div>"
        )

    html_parts.append("</div>")

# Champion column
champion = None
if "Final" in history and history["Final"]:
    champion = history["Final"][0]["winner"]
elif "Champion" in bracket:
    champion = bracket["Champion"]

html_parts.append(
    f"<div style='display:flex;flex-direction:column;justify-content:center;"
    f"flex:0.7;min-width:160px;padding:0 8px;'>"
    f"<div style='text-align:center;color:{muted};font-size:12px;text-transform:uppercase;"
    f"letter-spacing:1px;margin-bottom:12px;font-weight:600;'>Champion</div>"
)
if champion:
    html_parts.append(
        f"<div style='text-align:center;background:linear-gradient(135deg,rgba(53,208,127,.18),rgba(0,163,255,.10));"
        f"border:1.5px solid {accent};border-radius:14px;padding:28px 12px;"
        f"color:{accent};font-weight:800;font-size:20px;box-shadow:0 0 30px rgba(53,208,127,.18);'>"
        f"🏆<br>{champion}</div>"
    )
else:
    html_parts.append(
        f"<div style='text-align:center;border:1px dashed {muted};border-radius:14px;"
        f"padding:28px 12px;color:{muted};font-size:13px;'>TBD</div>"
    )
html_parts.append("</div>")
html_parts.append("</div>")

st.markdown("".join(html_parts), unsafe_allow_html=True)

# ── History table at the bottom ──────────────────────────────────────────────────
if not done_matches.empty:
    st.divider()
    st.markdown(f"<div class='section-title'>📋 Completed Match History</div>", unsafe_allow_html=True)
    display_history = done_matches[["date", "stage", "home_team", "home_goals", "away_goals", "away_team"]].copy()
    display_history.columns = ["Date", "Stage", "Home", "HG", "AG", "Away"]
    display_history = display_history.sort_values("Date", ascending=False)
    st.dataframe(display_history, use_container_width=True, hide_index=True)
