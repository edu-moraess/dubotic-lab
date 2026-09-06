"""
Tetris-style procedural pieces made of unit cubes.
No proprietary assets.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
import plotly.graph_objects as go


# Tetromino definitions (relative cell offsets)
TETROMINOS: Dict[str, List[Tuple[int, int]]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 2)],
    "J": [(1, 0), (1, 1), (1, 2), (0, 2)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
}

COLORS = {
    "I": "#00d4ff",
    "O": "#f6e05e",
    "T": "#b794f4",
    "L": "#ed8936",
    "J": "#63b3ed",
    "S": "#68d391",
    "Z": "#fc8181",
}


def _cell_edges(
    ox: float, oy: float, oz: float, cell: float = 25.0
) -> tuple:
    s = cell / 2
    verts = [
        (ox - s, oy - s, oz - s),
        (ox + s, oy - s, oz - s),
        (ox + s, oy + s, oz - s),
        (ox - s, oy + s, oz - s),
        (ox - s, oy - s, oz + s),
        (ox + s, oy - s, oz + s),
        (ox + s, oy + s, oz + s),
        (ox - s, oy + s, oz + s),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [verts[a][0], verts[b][0], None]
        ys += [verts[a][1], verts[b][1], None]
        zs += [verts[a][2], verts[b][2], None]
    return xs, ys, zs


class TetrisScene:
    name = "Tetris-style"
    description = "Procedural tetromino pieces for conceptual pick & place."

    def __init__(self):
        self.cell = 25.0
        # Place a few pieces in the workspace
        self.pieces = [
            ("T", (80.0, -100.0, 12.5)),
            ("L", (-90.0, 70.0, 12.5)),
            ("O", (130.0, 60.0, 12.5)),
            ("I", (-40.0, -120.0, 12.5)),
        ]

    def get_plotly_traces(self) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []
        for name, (px, py, pz) in self.pieces:
            color = COLORS.get(name, "#a0aec0")
            for dx, dy in TETROMINOS[name]:
                cx = px + dx * self.cell
                cy = py + dy * self.cell
                xs, ys, zs = _cell_edges(cx, cy, pz, self.cell)
                traces.append(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color=color, width=4),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        return traces

    def get_pick_place_targets(self) -> dict:
        # Pick the T piece, place near the O
        return {
            "pick": np.array([80.0 + self.cell, -100.0, 40.0]),
            "place": np.array([130.0, 60.0 + 2 * self.cell, 40.0]),
        }
