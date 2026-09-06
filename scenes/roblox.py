"""
Roblox-style procedural block world.
No proprietary assets – pure geometric cubes.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go


def _cube_edges(center: Tuple[float, float, float], size: float = 40.0) -> tuple:
    """Return x, y, z lists for the 12 edges of a cube (line segments)."""
    cx, cy, cz = center
    s = size / 2
    # 8 vertices
    verts = [
        (cx - s, cy - s, cz - s),
        (cx + s, cy - s, cz - s),
        (cx + s, cy + s, cz - s),
        (cx - s, cy + s, cz - s),
        (cx - s, cy - s, cz + s),
        (cx + s, cy - s, cz + s),
        (cx + s, cy + s, cz + s),
        (cx - s, cy + s, cz + s),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [verts[a][0], verts[b][0], None]
        ys += [verts[a][1], verts[b][1], None]
        zs += [verts[a][2], verts[b][2], None]
    return xs, ys, zs


class RobloxScene:
    name = "Roblox-style"
    description = "Procedural low-poly block world (grass, platforms, obstacles)."

    def __init__(self):
        self.block_size = 40.0
        # Procedural layout (no external assets)
        self.blocks = [
            # Grass platform under the robot
            ((0, 0, -20), "#48bb78"),
            ((40, 0, -20), "#48bb78"),
            ((-40, 0, -20), "#48bb78"),
            ((0, 40, -20), "#48bb78"),
            ((0, -40, -20), "#48bb78"),
            # Elevated platform
            ((120, -100, 0), "#ed8936"),
            ((160, -100, 0), "#ed8936"),
            ((120, -60, 0), "#ed8936"),
            # Wall / obstacle
            ((-120, 80, 20), "#a0aec0"),
            ((-120, 120, 20), "#a0aec0"),
            ((-80, 120, 20), "#a0aec0"),
            # Ramp approximation (stepped)
            ((80, 120, -10), "#ecc94b"),
            ((80, 160, 10), "#ecc94b"),
            # Target pedestal
            ((100, 80, 20), "#9f7aea"),
        ]

    def get_plotly_traces(self) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []
        for center, color in self.blocks:
            xs, ys, zs = _cube_edges(center, self.block_size)
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        return traces

    def get_pick_place_targets(self) -> dict:
        return {
            "pick": np.array([120.0, -80.0, 40.0]),
            "place": np.array([100.0, 80.0, 60.0]),
        }
