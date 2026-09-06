"""
Dubotic Lab — Vision layer (Hand Tracking)

Camera → Hand Tracking → Landmarks → Gestures → Coordinate Mapping → Target
"""

from .coordinate_mapping import CoordinateMapper, CameraBounds, WorkspaceBounds
from .smoothing import ExponentialSmoother
from .gestures import Gesture, GestureRecognizer, GestureState
from .hand_tracking import HandTracker, HandResult, LandmarkIndex
from .demo import SyntheticHandGenerator

__all__ = [
    "CoordinateMapper",
    "CameraBounds",
    "WorkspaceBounds",
    "ExponentialSmoother",
    "Gesture",
    "GestureRecognizer",
    "GestureState",
    "HandTracker",
    "HandResult",
    "LandmarkIndex",
    "SyntheticHandGenerator",
]
