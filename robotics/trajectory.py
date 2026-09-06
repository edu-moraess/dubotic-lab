"""
Joint-space trajectory planning with cubic polynomials.

For each joint:
  θ(t) = a0 + a1·t + a2·t² + a3·t³

Boundary conditions:
  θ(0) = θ_start,  θ̇(0) = 0
  θ(T) = θ_goal,   θ̇(T) = 0

Produces smooth position, velocity and acceleration profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .models import RobotModel


@dataclass
class Trajectory:
    """
    Discrete trajectory for all joints.

    time : (N,)
    position : (N, n_joints)
    velocity : (N, n_joints)
    acceleration : (N, n_joints)
    """

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    duration: float
    path_length: float  # approximate joint-space path length (rad)

    @property
    def n_points(self) -> int:
        return len(self.time)

    @property
    def max_velocity(self) -> float:
        return float(np.max(np.abs(self.velocity)))

    @property
    def max_acceleration(self) -> float:
        return float(np.max(np.abs(self.acceleration)))


class TrajectoryPlanner:
    """Cubic-polynomial joint-space trajectory generator."""

    def __init__(self, model: RobotModel, default_duration: float = 2.0):
        self.model = model
        self.default_duration = default_duration

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        duration: Optional[float] = None,
        n_points: int = 100,
    ) -> Trajectory:
        """
        Generate a smooth joint-space trajectory.

        Parameters
        ----------
        start, goal : array-like (n_joints,)
        duration : total time in seconds (None → default)
        n_points : number of discrete samples
        """
        start = np.asarray(start, dtype=float).ravel()
        goal = np.asarray(goal, dtype=float).ravel()
        if start.shape != goal.shape:
            raise ValueError("start and goal must have the same shape")

        T = float(duration) if duration is not None else self.default_duration
        if T <= 0:
            raise ValueError("duration must be positive")

        n_joints = len(start)
        t = np.linspace(0.0, T, n_points)

        # Pre-allocate
        pos = np.zeros((n_points, n_joints))
        vel = np.zeros((n_points, n_joints))
        acc = np.zeros((n_points, n_joints))

        for j in range(n_joints):
            a0, a1, a2, a3 = self._cubic_coeffs(start[j], goal[j], T)
            # θ(t) = a0 + a1 t + a2 t² + a3 t³
            # θ̇   = a1 + 2 a2 t + 3 a3 t²
            # θ̈   = 2 a2 + 6 a3 t
            pos[:, j] = a0 + a1 * t + a2 * t**2 + a3 * t**3
            vel[:, j] = a1 + 2 * a2 * t + 3 * a3 * t**2
            acc[:, j] = 2 * a2 + 6 * a3 * t

        # Approximate path length in joint space
        diffs = np.diff(pos, axis=0)
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

        return Trajectory(
            time=t,
            position=pos,
            velocity=vel,
            acceleration=acc,
            duration=T,
            path_length=path_len,
        )

    @staticmethod
    def _cubic_coeffs(
        theta0: float, thetaf: float, T: float
    ) -> tuple[float, float, float, float]:
        """
        Solve for cubic coefficients with zero start/end velocity.

        θ(0) = θ0, θ̇(0) = 0
        θ(T) = θf, θ̇(T) = 0
        """
        a0 = theta0
        a1 = 0.0
        a2 = 3.0 * (thetaf - theta0) / (T**2)
        a3 = -2.0 * (thetaf - theta0) / (T**3)
        return a0, a1, a2, a3
