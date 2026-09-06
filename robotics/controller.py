"""
PID controller with basic anti-windup and output saturation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PIDController:
    """
    Discrete PID controller.

    u(t) = Kp·e + Ki·∫e dt + Kd·de/dt

    Features
    --------
    - Independent gains per channel (vectorised)
    - Integral anti-windup (clamping)
    - Output saturation
    """

    Kp: np.ndarray
    Ki: np.ndarray
    Kd: np.ndarray
    output_limits: tuple[float, float] = (-np.inf, np.inf)
    integral_limits: tuple[float, float] = (-np.inf, np.inf)

    # Internal state
    _integral: np.ndarray = field(init=False, repr=False)
    _prev_error: np.ndarray = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.Kp = np.atleast_1d(np.asarray(self.Kp, dtype=float))
        self.Ki = np.atleast_1d(np.asarray(self.Ki, dtype=float))
        self.Kd = np.atleast_1d(np.asarray(self.Kd, dtype=float))
        n = len(self.Kp)
        self._integral = np.zeros(n)
        self._prev_error = np.zeros(n)

    @classmethod
    def scalar(
        cls,
        kp: float,
        ki: float,
        kd: float,
        output_limits: tuple[float, float] = (-np.inf, np.inf),
    ) -> "PIDController":
        return cls(
            Kp=np.array([kp]),
            Ki=np.array([ki]),
            Kd=np.array([kd]),
            output_limits=output_limits,
        )

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._prev_error[:] = 0.0
        self._initialized = False

    def update(
        self,
        target: np.ndarray,
        measurement: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute control signal.

        Parameters
        ----------
        target, measurement : array-like
        dt : time step (seconds) > 0
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        target = np.atleast_1d(np.asarray(target, dtype=float))
        measurement = np.atleast_1d(np.asarray(measurement, dtype=float))
        error = target - measurement

        # Proportional
        p = self.Kp * error

        # Integral with anti-windup (clamp)
        self._integral += error * dt
        lo_i, hi_i = self.integral_limits
        self._integral = np.clip(self._integral, lo_i, hi_i)
        i = self.Ki * self._integral

        # Derivative
        if not self._initialized:
            d = np.zeros_like(error)
            self._initialized = True
        else:
            d = self.Kd * (error - self._prev_error) / dt
        self._prev_error = error.copy()

        u = p + i + d

        # Output saturation
        lo, hi = self.output_limits
        u = np.clip(u, lo, hi)
        return u
