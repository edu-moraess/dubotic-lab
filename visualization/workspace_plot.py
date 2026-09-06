"""
Workspace sampling and visualization.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import plotly.graph_objects as go

from robotics.models import RobotModel
from robotics.kinematics import forward_kinematics


def sample_workspace(
    model: RobotModel,
    n_samples_per_joint: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """
    Efficient uniform sampling of joint space → end-effector cloud.

    Returns
    -------
    points : (N, 3) reachable positions in mm
    """
    rng = np.random.default_rng(seed)
    limits = model.joint_limits
    samples = []
    for lo, hi in limits:
        samples.append(rng.uniform(lo, hi, n_samples_per_joint))

    # Cartesian product via meshgrid for small n
    grids = np.meshgrid(*samples, indexing="ij")
    joint_configs = np.stack([g.ravel() for g in grids], axis=1)

    positions = []
    for q in joint_configs:
        fk = forward_kinematics(model, q)
        positions.append(fk.end_effector_position)
    return np.asarray(positions)


def create_workspace_figure(
    model: RobotModel,
    workspace_points: np.ndarray,
    current_ee: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> go.Figure:
    """Scatter cloud of reachable workspace + current EE + target."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=workspace_points[:, 0],
            y=workspace_points[:, 1],
            z=workspace_points[:, 2],
            mode="markers",
            marker=dict(size=2, color="#4a5568", opacity=0.35),
            name="Workspace",
            hoverinfo="skip",
        )
    )

    if current_ee is not None:
        p = np.asarray(current_ee)
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="markers",
                marker=dict(size=8, color="#00d4ff", symbol="diamond"),
                name="Current EE",
            )
        )

    if target is not None:
        t = np.asarray(target)
        fig.add_trace(
            go.Scatter3d(
                x=[t[0]],
                y=[t[1]],
                z=[t[2]],
                mode="markers",
                marker=dict(size=8, color="#68d391", symbol="x"),
                name="Target",
            )
        )

    max_r = model.max_reach * 1.1
    fig.update_layout(
        title="Reachable Workspace",
        scene=dict(
            xaxis=dict(range=[-max_r, max_r], backgroundcolor="#0e1117", gridcolor="#2d3748"),
            yaxis=dict(range=[-max_r, max_r], backgroundcolor="#0e1117", gridcolor="#2d3748"),
            zaxis=dict(range=[-20, max_r], backgroundcolor="#0e1117", gridcolor="#2d3748"),
            aspectmode="cube",
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=360,
    )
    return fig
