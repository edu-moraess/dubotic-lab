"""Safety gates for the camera-to-robot hand tracking pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .coordinate_mapping import WorkspaceBounds


@dataclass(frozen=True)
class SafetyDecision:
    """Result of a safety gate; ``allow_motion`` must be false to hold."""

    allow_motion: bool
    reason: str


def valid_landmarks(landmarks: np.ndarray) -> bool:
    """Return whether a MediaPipe-style landmark array is usable."""
    try:
        points = np.asarray(landmarks, dtype=float)
    except (TypeError, ValueError):
        return False
    return points.ndim == 2 and points.shape[0] == 21 and points.shape[1] >= 2 and bool(
        np.all(np.isfinite(points))
    )


def target_in_workspace(target: np.ndarray, bounds: WorkspaceBounds, tolerance: float = 1e-9) -> bool:
    """Check XYZ against configured workspace bounds, including fixed Z."""
    try:
        point = np.asarray(target, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return False
    return bool(
        bounds.x_min - tolerance <= point[0] <= bounds.x_max + tolerance
        and bounds.y_min - tolerance <= point[1] <= bounds.y_max + tolerance
        and abs(point[2] - bounds.z) <= tolerance
    )


def evaluate_motion(
    *,
    hand_present: bool,
    landmarks: Optional[np.ndarray],
    target: Optional[np.ndarray],
    workspace: WorkspaceBounds,
    ik_success: bool,
    stop_gesture: bool = False,
) -> SafetyDecision:
    """Return a conservative decision for the next robot command."""
    if stop_gesture:
        return SafetyDecision(False, "STOP gesture")
    if not hand_present:
        return SafetyDecision(False, "hand not detected")
    if landmarks is None or not valid_landmarks(landmarks):
        return SafetyDecision(False, "invalid landmarks")
    if target is None or not target_in_workspace(target, workspace):
        return SafetyDecision(False, "target outside workspace")
    if not ik_success:
        return SafetyDecision(False, "IK failed")
    return SafetyDecision(True, "safe")
