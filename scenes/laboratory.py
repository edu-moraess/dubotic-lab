"""
Standard laboratory scene – technical grid floor, axes, simple props.
"""

from __future__ import annotations

from typing import List
import numpy as np
import plotly.graph_objects as go


class LaboratoryScene:
    name = "Laboratory"
    description = "Clean technical laboratory with grid floor and reference axes."

    def get_plotly_traces(self) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []

        # Grid floor
        grid_size = 400
        step = 50
        for v in range(-grid_size, grid_size + 1, step):
            # lines parallel to X
            traces.append(
                go.Scatter3d(
                    x=[-grid_size, grid_size],
                    y=[v, v],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="#2d3748", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # lines parallel to Y
            traces.append(
                go.Scatter3d(
                    x=[v, v],
                    y=[-grid_size, grid_size],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="#2d3748", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Simple workbench (box outline)
        bx, by, bz = 120, -180, 0
        w, d, h = 80, 40, 30
        # top rectangle
        xs = [bx - w / 2, bx + w / 2, bx + w / 2, bx - w / 2, bx - w / 2]
        ys = [by - d / 2, by - d / 2, by + d / 2, by + d / 2, by - d / 2]
        zs = [h, h, h, h, h]
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="#718096", width=3),
                name="Workbench",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        return traces

    def get_pick_place_targets(self) -> dict:
        """Example pick & place locations for the laboratory."""
        return {
            "pick": np.array([120.0, -80.0, 40.0]),
            "place": np.array([80.0, 100.0, 50.0]),
        }
