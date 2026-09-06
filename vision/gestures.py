"""
Geometric gesture recognition from MediaPipe-style 21 hand landmarks.

Gestures
--------
MOVE    — index extended, other fingers folded
GRAB    — pinch (thumb tip close to index tip)
RELEASE — open hand (fingers extended)
STOP    — closed fist
UNKNOWN — otherwise

Landmarks follow MediaPipe Hands ordering (0–20).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class Gesture(Enum):
    UNKNOWN = auto()
    MOVE = auto()
    GRAB = auto()
    RELEASE = auto()
    STOP = auto()


# MediaPipe Hands landmark indices
class LM:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class GestureConfig:
    """Configurable thresholds (normalised image coordinates ≈ 0–1)."""

    pinch_threshold: float = 0.05
    finger_extend_ratio: float = 1.20  # tip-to-mcp / pip-to-mcp heuristic
    fist_tip_distance: float = 0.12


@dataclass
class GestureState:
    gesture: Gesture = Gesture.UNKNOWN
    confidence: float = 0.0
    pinch_distance: float = 1.0


class GestureRecognizer:
    """Rule-based gesture classifier from 21 landmarks."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()

    def recognize(self, landmarks: np.ndarray) -> GestureState:
        """
        Parameters
        ----------
        landmarks : (21, 2) or (21, 3) array of normalised coordinates
        """
        lm = np.asarray(landmarks, dtype=float)
        if lm.ndim != 2 or lm.shape[0] != 21:
            return GestureState(Gesture.UNKNOWN, 0.0)

        pts = lm[:, :2]  # use x,y only

        pinch = self._distance(pts[LM.THUMB_TIP], pts[LM.INDEX_TIP])
        index_ext = self._is_extended(pts, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP)
        middle_ext = self._is_extended(pts, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP)
        ring_ext = self._is_extended(pts, LM.RING_MCP, LM.RING_PIP, LM.RING_TIP)
        pinky_ext = self._is_extended(pts, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP)

        n_extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # Priority: fist / point / open / pinch
        if n_extended == 0:
            # closed hand — distinguish fist vs pinch by thumb-index distance
            if pinch < self.config.pinch_threshold * 0.6:
                return GestureState(Gesture.GRAB, confidence=0.9, pinch_distance=pinch)
            return GestureState(Gesture.STOP, confidence=0.85, pinch_distance=pinch)

        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return GestureState(Gesture.MOVE, confidence=0.9, pinch_distance=pinch)

        if pinch < self.config.pinch_threshold:
            return GestureState(Gesture.GRAB, confidence=0.9, pinch_distance=pinch)

        if n_extended >= 3:
            return GestureState(Gesture.RELEASE, confidence=0.85, pinch_distance=pinch)

        return GestureState(Gesture.UNKNOWN, confidence=0.3, pinch_distance=pinch)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _is_extended(
        self,
        pts: np.ndarray,
        mcp: int,
        pip: int,
        tip: int,
    ) -> bool:
        """
        Finger is extended if tip is sufficiently far from MCP
        relative to a reference hand scale (wrist–middle-MCP distance).
        """
        # hand scale
        scale = self._distance(pts[0], pts[9])  # wrist to middle MCP
        if scale < 1e-6:
            scale = 0.1
        d_tip = self._distance(pts[tip], pts[mcp])
        # extended if tip is at least ~55% of a typical full finger length (~1.2*scale)
        return d_tip > self.config.finger_extend_ratio * scale
