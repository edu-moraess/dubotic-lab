"""
Pixel → workspace coordinate mapping.

Maps camera image coordinates (u, v) into robot workspace (X, Y)
on a fixed working plane (Z = constant).

This is an affine mapping defined by configurable bounds.
It is NOT a full camera calibration or true metric 3D reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class CameraBounds:
    """Pixel bounds of the region of interest in the camera image."""

    u_min: float = 0.0
    u_max: float = 640.0
    v_min: float = 0.0
    v_max: float = 480.0

    def clamp(self, u: float, v: float) -> Tuple[float, float]:
        return (
            float(np.clip(u, self.u_min, self.u_max)),
            float(np.clip(v, self.v_min, self.v_max)),
        )


@dataclass(frozen=True)
class WorkspaceBounds:
    """
    Robot workspace bounds in millimetres corresponding to camera bounds.

    Note the typical image v-axis points downward while robot Y often
    points "forward"; the mapper supports flipping via invert_v.
    """

    x_min: float = -200.0
    x_max: float = 200.0
    y_min: float = -150.0
    y_max: float = 150.0
    z: float = 50.0  # fixed working-plane height (mm)


class CoordinateMapper:
    """
    Affine pixel → workspace mapper.

    Mapping (with optional vertical flip):
        nx = (u - u_min) / (u_max - u_min)          ∈ [0, 1]
        ny = (v - v_min) / (v_max - v_min)          ∈ [0, 1]
        if invert_v: ny = 1 - ny
        X = x_min + nx * (x_max - x_min)
        Y = y_min + ny * (y_max - y_min)
        Z = workspace.z
    """

    def __init__(
        self,
        camera: CameraBounds | None = None,
        workspace: WorkspaceBounds | None = None,
        invert_v: bool = True,
    ):
        self.camera = camera or CameraBounds()
        self.workspace = workspace or WorkspaceBounds()
        self.invert_v = invert_v

    def pixel_to_workspace(
        self, u: float, v: float
    ) -> np.ndarray:
        """
        Convert pixel coordinates to workspace XYZ (mm).

        Out-of-bounds pixels are clamped to camera bounds.
        """
        if not np.isfinite(u) or not np.isfinite(v):
            raise ValueError(f"Invalid pixel coordinates: u={u}, v={v}")

        u_c, v_c = self.camera.clamp(u, v)

        du = self.camera.u_max - self.camera.u_min
        dv = self.camera.v_max - self.camera.v_min
        if du <= 0 or dv <= 0:
            raise ValueError("Camera bounds must have positive width/height")

        nx = (u_c - self.camera.u_min) / du
        ny = (v_c - self.camera.v_min) / dv
        if self.invert_v:
            ny = 1.0 - ny

        ws = self.workspace
        x = ws.x_min + nx * (ws.x_max - ws.x_min)
        y = ws.y_min + ny * (ws.y_max - ws.y_min)
        z = ws.z
        return np.array([x, y, z], dtype=float)

    def workspace_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Inverse mapping (for visualisation / debugging)."""
        ws = self.workspace
        dx = ws.x_max - ws.x_min
        dy = ws.y_max - ws.y_min
        if dx <= 0 or dy <= 0:
            raise ValueError("Workspace bounds must have positive extent")

        nx = (x - ws.x_min) / dx
        ny = (y - ws.y_min) / dy
        if self.invert_v:
            ny = 1.0 - ny

        cam = self.camera
        u = cam.u_min + nx * (cam.u_max - cam.u_min)
        v = cam.v_min + ny * (cam.v_max - cam.v_min)
        return float(u), float(v)
