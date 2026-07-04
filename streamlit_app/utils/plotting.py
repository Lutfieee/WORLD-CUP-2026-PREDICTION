"""Interactive football analytics visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_TEMPLATE = "plotly_dark"


def radar_chart(frame: pd.DataFrame, label_col: str, metrics: list[str], title: str) -> go.Figure:
    """Create a normalized radar chart for team or player comparison."""

    fig = go.Figure()
    for _, row in frame.iterrows():
        values = []
        for metric in metrics:
            max_value = max(float(frame[metric].max()), 1e-9)
            values.append(float(row[metric]) / max_value)
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=str(row[label_col]),
            )
        )
    fig.update_layout(template=PLOT_TEMPLATE, title=title, polar=dict(radialaxis=dict(visible=False)), height=430)
    return fig


def probability_bar(row: pd.Series) -> go.Figure:
    """Create a horizontal bar chart for match outcome probabilities."""

    labels = ["Home Win", "Draw", "Away Win"]
    values = [row.get("prob_home_win", 0), row.get("prob_draw", 0), row.get("prob_away_win", 0)]
    fig = px.bar(x=values, y=labels, orientation="h", text=[f"{v:.0%}" for v in values], template=PLOT_TEMPLATE)
    fig.update_traces(marker_color=["#35d07f", "#f2c94c", "#ff6b6b"], textposition="outside")
    fig.update_layout(xaxis_tickformat=".0%", height=280, margin=dict(l=10, r=30, t=10, b=10))
    return fig


def pitch_shot_map(seed: int = 7, goals_only: bool = False) -> go.Figure:
    """Generate an interactive synthetic shot map on a football pitch."""

    rng = np.random.default_rng(seed)
    shots = pd.DataFrame(
        {
            "x": np.clip(rng.normal(82, 11, 85), 48, 100),
            "y": np.clip(rng.normal(40, 18, 85), 2, 78),
            "xg": np.clip(rng.beta(1.6, 7.5, 85), 0.01, 0.75),
        }
    )
    shots["goal"] = rng.random(len(shots)) < shots["xg"]
    if goals_only:
        shots = shots[shots["goal"]]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=80, line=dict(color="#9aa4b2", width=2))
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=80, line=dict(color="#687486", width=1))
    fig.add_shape(type="rect", x0=82, y0=18, x1=100, y1=62, line=dict(color="#687486", width=1))
    fig.add_shape(type="rect", x0=94, y0=30, x1=100, y1=50, line=dict(color="#687486", width=1))
    fig.add_trace(
        go.Scatter(
            x=shots["x"],
            y=shots["y"],
            mode="markers",
            marker=dict(size=shots["xg"] * 42 + 8, color=shots["xg"], colorscale="Viridis", showscale=True),
            text=[f"xG {value:.2f}" for value in shots["xg"]],
            name="Shots",
        )
    )
    fig.update_layout(template=PLOT_TEMPLATE, height=520, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=5, r=5, t=20, b=5))
    return fig


def xg_timeline() -> go.Figure:
    """Create a synthetic cumulative xG flow chart."""

    minutes = np.array([4, 12, 18, 26, 34, 43, 51, 59, 67, 73, 82, 90])
    home = np.cumsum([0.03, 0.08, 0.22, 0.05, 0.31, 0.04, 0.11, 0.42, 0.09, 0.14, 0.56, 0.07])
    away = np.cumsum([0.02, 0.15, 0.04, 0.28, 0.06, 0.10, 0.21, 0.05, 0.12, 0.35, 0.04, 0.10])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=minutes, y=home, mode="lines+markers", name="Home xG", line=dict(color="#35d07f")))
    fig.add_trace(go.Scatter(x=minutes, y=away, mode="lines+markers", name="Away xG", line=dict(color="#ff6b6b")))
    fig.update_layout(template=PLOT_TEMPLATE, title="Cumulative xG Flow", height=430, xaxis_title="Minute", yaxis_title="xG")
    return fig


def heatmap_matrix(frame: pd.DataFrame, x: str, y: str, value: str, title: str) -> go.Figure:
    """Create a ranked heatmap for tournament analytics."""

    pivot = frame.pivot_table(index=y, columns=x, values=value, aggfunc="mean").fillna(0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", template=PLOT_TEMPLATE, title=title)
    fig.update_layout(height=430)
    return fig
