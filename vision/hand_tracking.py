"""
Hand landmark detection.

Primary backend: MediaPipe Hands (when available).
Fallback: None detection (caller may use SyntheticHandGenerator).

Does not depend on a live camera — accepts an RGB image array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple
import numpy as np


class LandmarkIndex(IntEnum):
    """MediaPipe Hands landmark indices."""

    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20


@dataclass
class HandResult:
    """Single detected hand."""

    landmarks: np.ndarray  # (21, 3) normalised [x, y, z_rel]
    landmarks_px: np.ndarray  # (21, 2) pixel coordinates
    handedness: str  # "Left" / "Right"
    confidence: float
    image_size: Tuple[int, int]  # (width, height)

    @property
    def index_tip_px(self) -> np.ndarray:
        return self.landmarks_px[LandmarkIndex.INDEX_TIP]

    @property
    def index_tip_norm(self) -> np.ndarray:
        return self.landmarks[LandmarkIndex.INDEX_TIP, :2]


@dataclass
class TrackerOutput:
    hands: List[HandResult] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None  # RGB with drawings

    @property
    def primary(self) -> Optional[HandResult]:
        return self.hands[0] if self.hands else None

    @property
    def has_hand(self) -> bool:
        return len(self.hands) > 0


class HandTracker:
    """
    Detects up to `max_hands` hands in an RGB image.

    Parameters
    ----------
    max_hands : int
    min_detection_confidence : float
    min_tracking_confidence : float
    draw : bool  — whether to return annotated image
    """

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        draw: bool = True,
    ):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.draw = draw
        self._mp_hands = None
        self._mp_drawing = None
        self._hands = None
        self._backend = "none"
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp

            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._backend = "mediapipe"
        except Exception:
            self._backend = "none"
            self._hands = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def available(self) -> bool:
        return self._backend == "mediapipe"

    def process(self, image_rgb: np.ndarray) -> TrackerOutput:
        """
        Process one RGB frame (H, W, 3), dtype uint8.

        Returns TrackerOutput (possibly empty).
        """
        if image_rgb is None or image_rgb.size == 0:
            return TrackerOutput()

        if self._hands is None:
            return TrackerOutput()

        h, w = image_rgb.shape[:2]
        results = self._hands.process(image_rgb)

        hands: List[HandResult] = []
        annotated = image_rgb.copy() if self.draw else None

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                lm_norm = np.zeros((21, 3), dtype=float)
                lm_px = np.zeros((21, 2), dtype=float)
                for idx, lm in enumerate(hand_lms.landmark):
                    lm_norm[idx] = [lm.x, lm.y, lm.z]
                    lm_px[idx] = [lm.x * w, lm.y * h]

                label = "Right"
                conf = 0.0
                if i < len(handedness_list):
                    label = handedness_list[i].classification[0].label
                    conf = handedness_list[i].classification[0].score

                hands.append(
                    HandResult(
                        landmarks=lm_norm,
                        landmarks_px=lm_px,
                        handedness=label,
                        confidence=float(conf),
                        image_size=(w, h),
                    )
                )

                if annotated is not None and self._mp_drawing is not None:
                    self._mp_drawing.draw_landmarks(
                        annotated,
                        hand_lms,
                        self._mp_hands.HAND_CONNECTIONS,
                    )

        return TrackerOutput(hands=hands, annotated_image=annotated)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
