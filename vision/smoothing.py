"""
Signal smoothing for hand-tracking targets.

Exponential Moving Average (EMA) reduces landmark jitter before
the position is sent to Inverse Kinematics.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


class ExponentialSmoother:
    """
    First-order low-pass filter (EMA).

    y[k] = α · x[k] + (1 − α) · y[k−1]

    α ∈ (0, 1]:
        α → 1  → almost no smoothing (reactive)
        α → 0  → heavy smoothing (laggy)
    """

    def __init__(self, alpha: float = 0.35, dim: int = 3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.dim = dim
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def update(self, sample: np.ndarray) -> np.ndarray:
        """Filter one sample. Returns smoothed value."""
        sample = np.asarray(sample, dtype=float).ravel()
        if sample.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {sample.shape[0]}")
        if not np.all(np.isfinite(sample)):
            # Reject non-finite samples; keep previous state if available
            if self._state is not None:
                return self._state.copy()
            raise ValueError("Non-finite sample and no previous state")

        if self._state is None:
            self._state = sample.copy()
        else:
            self._state = self.alpha * sample + (1.0 - self.alpha) * self._state
        return self._state.copy()

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()
