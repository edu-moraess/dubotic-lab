"""
Trajectory metric plots (joint position / velocity / acceleration vs time).
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from robotics.trajectory import Trajectory


def create_trajectory_plots(traj: Trajectory) -> go.Figure:
    """Three stacked plots: position, velocity, acceleration."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Joint Position (rad)", "Joint Velocity (rad/s)", "Joint Acceleration (rad/s²)"),
        vertical_spacing=0.08,
    )

    colors = ["#00d4ff", "#f6ad55", "#fc8181"]
    names = ["J1", "J2", "J3"]

    for j in range(traj.position.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=traj.time,
                y=traj.position[:, j],
                name=names[j],
                line=dict(color=colors[j]),
                legendgroup=names[j],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=traj.time,
                y=traj.velocity[:, j],
                name=names[j],
                line=dict(color=colors[j]),
                legendgroup=names[j],
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=traj.time,
                y=traj.acceleration[:, j],
                name=names[j],
                line=dict(color=colors[j]),
                legendgroup=names[j],
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=520,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1, gridcolor="#2d3748")
    for r in (1, 2, 3):
        fig.update_yaxes(gridcolor="#2d3748", row=r, col=1)
        fig.update_xaxes(gridcolor="#2d3748", row=r, col=1)

    return fig
