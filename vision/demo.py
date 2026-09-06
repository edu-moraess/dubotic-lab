"""
Synthetic hand landmark generator for Demo / Simulation mode.

Produces MediaPipe-compatible (21, 3) landmarks without a camera.
Explicitly labelled as SIMULATION — never presented as real camera data.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from .hand_tracking import HandResult


class SyntheticHandGenerator:
    """Generates controllable synthetic hands for pipeline testing."""

    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        self.image_size = image_size
        self._t = 0.0

    def generate(
        self,
        center_norm: Tuple[float, float] = (0.5, 0.5),
        gesture: str = "MOVE",
        scale: float = 0.25,
        noise: float = 0.0,
        seed: int | None = None,
    ) -> HandResult:
        rng = np.random.default_rng(seed)
        w, h = self.image_size
        cx, cy = center_norm

        g = gesture.upper()
        if g == "GRAB":
            lm = self._template_grab()
        elif g == "STOP":
            lm = self._template_fist()
        elif g == "MOVE":
            lm = self._template_point()
        else:
            lm = self._template_open()

        lm = lm * scale
        lm[:, 0] += cx
        lm[:, 1] += cy
        if noise > 0:
            lm[:, :2] += rng.normal(0, noise, size=(21, 2))
        lm = np.clip(lm, 0.0, 1.0)

        lm_px = lm[:, :2].copy()
        lm_px[:, 0] *= w
        lm_px[:, 1] *= h

        return HandResult(
            landmarks=lm,
            landmarks_px=lm_px,
            handedness="Right",
            confidence=1.0,
            image_size=(w, h),
        )

    def generate_moving(
        self,
        t: float,
        radius: float = 0.2,
        gesture: str = "MOVE",
    ) -> HandResult:
        cx = 0.5 + radius * np.cos(t)
        cy = 0.5 + radius * np.sin(t * 0.7)
        return self.generate(center_norm=(cx, cy), gesture=gesture, noise=0.003)

    # ------------------------------------------------------------------
    @staticmethod
    def _base() -> np.ndarray:
        lm = np.zeros((21, 3))
        lm[0] = [0.0, 0.20, 0.0]  # wrist
        # thumb chain
        lm[1] = [-0.10, 0.12, 0.0]
        lm[2] = [-0.18, 0.05, 0.0]
        lm[3] = [-0.24, -0.02, 0.0]
        lm[4] = [-0.30, -0.10, 0.0]
        # index
        lm[5] = [-0.05, 0.0, 0.0]
        lm[6] = [-0.05, -0.12, 0.0]
        lm[7] = [-0.05, -0.22, 0.0]
        lm[8] = [-0.05, -0.35, 0.0]
        # middle
        lm[9] = [0.03, 0.0, 0.0]
        lm[10] = [0.03, -0.14, 0.0]
        lm[11] = [0.03, -0.26, 0.0]
        lm[12] = [0.03, -0.38, 0.0]
        # ring
        lm[13] = [0.10, 0.02, 0.0]
        lm[14] = [0.10, -0.10, 0.0]
        lm[15] = [0.10, -0.20, 0.0]
        lm[16] = [0.10, -0.30, 0.0]
        # pinky
        lm[17] = [0.16, 0.05, 0.0]
        lm[18] = [0.17, -0.04, 0.0]
        lm[19] = [0.18, -0.12, 0.0]
        lm[20] = [0.19, -0.20, 0.0]
        return lm

    @classmethod
    def _template_open(cls) -> np.ndarray:
        return cls._base()

    @classmethod
    def _template_point(cls) -> np.ndarray:
        """Index extended; other fingers folded; thumb far from index."""
        lm = cls._base()
        # fold middle/ring/pinky: tip near MCP
        for mcp, pip, tip in [(9, 10, 12), (13, 14, 16), (17, 18, 20)]:
            lm[pip] = lm[mcp] + np.array([0.0, -0.03, 0.0])
            lm[tip] = lm[mcp] + np.array([0.0, -0.04, 0.0])
        # thumb away
        lm[4] = lm[8] + np.array([-0.25, 0.15, 0.0])
        return lm

    @classmethod
    def _template_grab(cls) -> np.ndarray:
        lm = cls._base()
        lm[4] = lm[8] + np.array([0.01, 0.01, 0.0])  # pinch
        return lm

    @classmethod
    def _template_fist(cls) -> np.ndarray:
        lm = cls._base()
        # all tips near palm; thumb clearly separated from index
        lm[4] = lm[0] + np.array([-0.18, 0.02, 0.0])
        for tip in [8, 12, 16, 20]:
            lm[tip] = lm[0] + np.array([0.03 + 0.02 * (tip // 4), -0.05, 0.0])
        for pip in [6, 10, 14, 18]:
            lm[pip] = lm[0] + np.array([0.02, -0.03, 0.0])
        return lm
