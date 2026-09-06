"""
3D robot visualization with Plotly.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import plotly.graph_objects as go

from robotics.models import RobotModel
from robotics.kinematics import forward_kinematics


# Color palette – technical / cyber
COLORS = {
    "base": "#4a5568",
    "link": "#00d4ff",
    "joint": "#f6ad55",
    "ee": "#fc8181",
    "target": "#68d391",
    "trajectory": "#b794f4",
    "axis_x": "#fc8181",
    "axis_y": "#68d391",
    "axis_z": "#63b3ed",
    "grid": "#2d3748",
}


def _axis_traces(length: float = 80.0) -> list:
    """Coordinate frame at origin."""
    traces = []
    for axis, color, name in [
        ([length, 0, 0], COLORS["axis_x"], "X"),
        ([0, length, 0], COLORS["axis_y"], "Y"),
        ([0, 0, length], COLORS["axis_z"], "Z"),
    ]:
        traces.append(
            go.Scatter3d(
                x=[0, axis[0]],
                y=[0, axis[1]],
                z=[0, axis[2]],
                mode="lines",
                line=dict(color=color, width=4),
                name=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def create_robot_figure(
    model: RobotModel,
    joint_angles: np.ndarray,
    target: Optional[np.ndarray] = None,
    trajectory_points: Optional[np.ndarray] = None,
    scene_objects: Optional[Sequence[go.Scatter3d]] = None,
    title: str = "Dubotic Lab – 3-DOF Arm",
) -> go.Figure:
    """
    Build a complete 3D figure of the robot + optional target / trajectory / scene.
    """
    fk = forward_kinematics(model, joint_angles)
    pts = fk.joint_positions  # (4, 3)  base, J2, J3, EE

    fig = go.Figure()

    # Axes
    for tr in _axis_traces():
        fig.add_trace(tr)

    # Base platform (simple disk approximation via scatter)
    theta = np.linspace(0, 2 * np.pi, 24)
    r_base = 40.0
    fig.add_trace(
        go.Scatter3d(
            x=r_base * np.cos(theta),
            y=r_base * np.sin(theta),
            z=np.zeros_like(theta),
            mode="lines",
            line=dict(color=COLORS["base"], width=6),
            name="Base",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Links
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines+markers",
            line=dict(color=COLORS["link"], width=10),
            marker=dict(
                size=[10, 12, 12, 14],
                color=[COLORS["base"], COLORS["joint"], COLORS["joint"], COLORS["ee"]],
                symbol=["circle", "circle", "circle", "diamond"],
            ),
            name="Robot",
            hovertemplate="(%{x:.1f}, %{y:.1f}, %{z:.1f}) mm<extra></extra>",
        )
    )

    # Target
    if target is not None:
        t = np.asarray(target, dtype=float).ravel()
        fig.add_trace(
            go.Scatter3d(
                x=[t[0]],
                y=[t[1]],
                z=[t[2]],
                mode="markers",
                marker=dict(size=10, color=COLORS["target"], symbol="x", line=dict(width=2)),
                name="Target",
                hovertemplate="Target (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>",
            )
        )

    # Trajectory path (end-effector)
    if trajectory_points is not None and len(trajectory_points) > 1:
        tp = np.asarray(trajectory_points)
        fig.add_trace(
            go.Scatter3d(
                x=tp[:, 0],
                y=tp[:, 1],
                z=tp[:, 2],
                mode="lines",
                line=dict(color=COLORS["trajectory"], width=3, dash="dot"),
                name="Trajectory",
                hoverinfo="skip",
            )
        )

    # Scene objects (from Laboratory / Roblox / Tetris)
    if scene_objects:
        for obj in scene_objects:
            fig.add_trace(obj)

    # Layout – mobile friendly
    max_r = model.max_reach * 1.15
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        scene=dict(
            xaxis=dict(
                title="X (mm)",
                range=[-max_r, max_r],
                backgroundcolor="#0e1117",
                gridcolor=COLORS["grid"],
                showbackground=True,
            ),
            yaxis=dict(
                title="Y (mm)",
                range=[-max_r, max_r],
                backgroundcolor="#0e1117",
                gridcolor=COLORS["grid"],
                showbackground=True,
            ),
            zaxis=dict(
                title="Z (mm)",
                range=[-20, max_r + 50],
                backgroundcolor="#0e1117",
                gridcolor=COLORS["grid"],
                showbackground=True,
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.1),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def update_robot_figure(
    fig: go.Figure,
    model: RobotModel,
    joint_angles: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> go.Figure:
    """Lightweight update of robot + target traces (index 1 = robot, 2 = target if present)."""
    # For simplicity in Streamlit we rebuild; Plotly figure updates are more complex
    return create_robot_figure(model, joint_angles, target=target)
