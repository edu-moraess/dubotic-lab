"""
Robot model and state definitions.

All geometric and kinematic parameters are centralized here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class RobotModel:
    """
    Explicit geometric model of a serial robotic arm.

    Units: lengths in millimeters, angles in radians internally,
    joint limits stored in radians.
    """

    name: str
    link_lengths: Tuple[float, ...]  # L1, L2, L3, ... in mm
    joint_limits: Tuple[Tuple[float, float], ...]  # (min, max) per joint in rad
    joint_count: int

    def __post_init__(self):
        if len(self.link_lengths) != self.joint_count:
            raise ValueError(
                f"link_lengths length ({len(self.link_lengths)}) "
                f"must equal joint_count ({self.joint_count})"
            )
        if len(self.joint_limits) != self.joint_count:
            raise ValueError(
                f"joint_limits length ({len(self.joint_limits)}) "
                f"must equal joint_count ({self.joint_count})"
            )

    @classmethod
    def default_3dof(cls) -> "RobotModel":
        """
        Standard 3-DOF anthropomorphic-style arm used by Dubotic Lab.

        L1 = 150 mm (base to shoulder)
        L2 = 150 mm (upper arm)
        L3 = 100 mm (forearm / end-effector offset)

        Joint limits (degrees converted to radians):
        J1: -180° → +180°
        J2: -90°  → +90°
        J3: -135° → +135°
        """
        deg2rad = np.pi / 180.0
        return cls(
            name="Dubotic-3DOF",
            link_lengths=(150.0, 150.0, 100.0),
            joint_limits=(
                (-180.0 * deg2rad, 180.0 * deg2rad),
                (-90.0 * deg2rad, 90.0 * deg2rad),
                (-135.0 * deg2rad, 135.0 * deg2rad),
            ),
            joint_count=3,
        )

    def clamp_joints(self, angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to model limits."""
        clamped = np.asarray(angles, dtype=float).copy()
        for i, (lo, hi) in enumerate(self.joint_limits):
            clamped[i] = np.clip(clamped[i], lo, hi)
        return clamped

    def within_limits(self, angles: np.ndarray, tol: float = 1e-6) -> bool:
        """Check whether joint angles respect limits (with small tolerance)."""
        angles = np.asarray(angles, dtype=float)
        for i, (lo, hi) in enumerate(self.joint_limits):
            if angles[i] < lo - tol or angles[i] > hi + tol:
                return False
        return True

    @property
    def max_reach(self) -> float:
        """Maximum theoretical reach (sum of link lengths)."""
        return float(np.sum(self.link_lengths))

    @property
    def min_reach(self) -> float:
        """
        Approximate minimum reach for this planar-ish geometry.
        Conservative estimate used for workspace checks.
        """
        L1, L2, L3 = self.link_lengths
        return max(0.0, abs(L2 - L3) * 0.5)  # conservative


@dataclass
class RobotState:
    """
    Complete instantaneous state of the robot.

    joint_angles, velocities, accelerations: shape (n_joints,)
    end_effector_position: (x, y, z) in mm
    end_effector_orientation: 3x3 rotation matrix (world frame)
    """

    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    end_effector_position: np.ndarray
    end_effector_orientation: np.ndarray
    timestamp: float = 0.0

    def __post_init__(self):
        self.joint_angles = np.asarray(self.joint_angles, dtype=float)
        self.joint_velocities = np.asarray(self.joint_velocities, dtype=float)
        self.joint_accelerations = np.asarray(self.joint_accelerations, dtype=float)
        self.end_effector_position = np.asarray(self.end_effector_position, dtype=float)
        self.end_effector_orientation = np.asarray(
            self.end_effector_orientation, dtype=float
        )

    @classmethod
    def zeros(cls, n_joints: int = 3) -> "RobotState":
        return cls(
            joint_angles=np.zeros(n_joints),
            joint_velocities=np.zeros(n_joints),
            joint_accelerations=np.zeros(n_joints),
            end_effector_position=np.zeros(3),
            end_effector_orientation=np.eye(3),
            timestamp=0.0,
        )

    def copy(self) -> "RobotState":
        return RobotState(
            joint_angles=self.joint_angles.copy(),
            joint_velocities=self.joint_velocities.copy(),
            joint_accelerations=self.joint_accelerations.copy(),
            end_effector_position=self.end_effector_position.copy(),
            end_effector_orientation=self.end_effector_orientation.copy(),
            timestamp=self.timestamp,
        )
