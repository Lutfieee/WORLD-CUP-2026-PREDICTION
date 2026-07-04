"""Reusable Streamlit UI primitives."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def apply_page_config() -> None:
    """Configure the Streamlit page."""

    st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="WC", layout="wide")


def inject_css(theme: str = "dark") -> None:
    """Inject premium SaaS CSS with dark, light, and auto modes."""

    dark = theme in {"Dark", "Auto"}
    bg = "#071016" if dark else "#f6f8fb"
    panel = "rgba(255,255,255,0.08)" if dark else "rgba(255,255,255,0.94)"
    text = "#f6fbff" if dark else "#182230"
    muted = "#9aa4b2" if dark else "#667085"
    band = "#0e1924" if dark else "#eef4f7"
    table_bg = "#0b1118" if dark else "#ffffff"
    table_text = "#f6fbff" if dark else "#182230"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 15% 5%, rgba(53,208,127,.16), transparent 24%),
                linear-gradient(135deg, {bg} 0%, {band} 100%);
            color: {text};
        }}
        [data-testid="stSidebar"] {{
            background: {"rgba(4,10,16,.92)" if dark else "rgba(255,255,255,.86)"};
            border-right: 1px solid rgba(255,255,255,.10);
        }}
        .metric-card {{
            background: {panel};
            border: 1px solid {"rgba(255,255,255,.12)" if dark else "rgba(16,24,40,.10)"};
            border-radius: 8px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 18px 60px rgba(0,0,0,.22);
            backdrop-filter: blur(14px);
            min-height: 116px;
        }}
        .metric-label {{
            color: {muted};
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
        }}
        .metric-value {{
            color: {text};
            font-size: 30px;
            line-height: 1.1;
            font-weight: 780;
            margin-top: 8px;
        }}
        .section-title {{
            font-size: 22px;
            font-weight: 760;
            margin: 4px 0 12px 0;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {"rgba(255,255,255,.10)" if dark else "rgba(16,24,40,.12)"};
            border-radius: 8px;
            overflow: hidden;
            background: {table_bg};
            color: {table_text};
        }}
        div[data-testid="stDataFrame"] * {{
            color: {table_text};
        }}
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid rgba(53,208,127,.35);
            background: linear-gradient(135deg, #35d07f, #00a3ff);
            color: #061016;
            font-weight: 760;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_template(theme: str = "Dark") -> str:
    """Return a Plotly template matching the active app theme."""

    return "plotly_dark" if theme in {"Dark", "Auto"} else "plotly_white"


def kpi_card(label: str, value: str, hint: str = "") -> None:
    """Render a glass KPI card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div style="color:#9aa4b2;font-size:13px;margin-top:8px;">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_header(title: str, subtitle: str) -> None:
    """Render a page header."""

    st.markdown(f"<h1 style='margin-bottom:4px'>{title}</h1>", unsafe_allow_html=True)
    st.caption(subtitle)


def format_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Format probability columns for display."""

    display = frame.copy()
    for col in display.columns:
        if col.startswith("prob_") or col in {"confidence", "champion", "runner_up", "semi_final", "quarter_final"}:
            display[col] = display[col].map(lambda value: f"{float(value):.1%}")
    return display
