"""
Forward Kinematics for the 3-DOF Dubotic arm.

Geometry (standard serial chain):
  Base at origin.
  J1: rotation about Z (yaw) – base rotation.
  Link L1 along Z after J1 (vertical column).
  J2: rotation about Y (pitch) – shoulder.
  Link L2 in the XZ plane after J2.
  J3: rotation about Y (pitch) – elbow.
  Link L3 continues in the same plane.

This is a classic R-R-R anthropomorphic configuration
with the last two joints coplanar (pitch-pitch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .models import RobotModel, RobotState
from .transforms import (
    rotation_z,
    rotation_y,
    homogeneous_transform,
    compose,
)


@dataclass
class FKResult:
    """Complete forward-kinematics output."""

    joint_positions: np.ndarray  # (n_joints+1, 3)  base + each joint + EE
    end_effector_position: np.ndarray  # (3,)
    end_effector_orientation: np.ndarray  # (3, 3)
    transforms: list  # list of 4x4 from base to each frame
    T_ee: np.ndarray  # final 4x4


def forward_kinematics(
    model: RobotModel, joint_angles: np.ndarray
) -> FKResult:
    """
    Compute forward kinematics.

    Parameters
    ----------
    model : RobotModel
    joint_angles : array-like, shape (3,)  [θ1, θ2, θ3] in radians

    Returns
    -------
    FKResult
    """
    theta = np.asarray(joint_angles, dtype=float).ravel()
    if theta.shape[0] != model.joint_count:
        raise ValueError(
            f"Expected {model.joint_count} joint angles, got {theta.shape[0]}"
        )

    L1, L2, L3 = model.link_lengths

    # Frame 0 → 1 : rotate about Z by θ1, then translate along local Z by L1
    # T = Rot @ Trans  ⇒ translation expressed in the rotated frame
    R1 = rotation_z(theta[0])
    t1_local = np.array([0.0, 0.0, L1])
    T01 = homogeneous_transform(R1, R1 @ t1_local)

    # Frame 1 → 2 : rotate about Y by θ2, then translate along local X by L2
    R2 = rotation_y(theta[1])
    t2_local = np.array([L2, 0.0, 0.0])
    T12 = homogeneous_transform(R2, R2 @ t2_local)

    # Frame 2 → 3 : rotate about Y by θ3, then translate along local X by L3
    R3 = rotation_y(theta[2])
    t3_local = np.array([L3, 0.0, 0.0])
    T23 = homogeneous_transform(R3, R3 @ t3_local)

    # Compose
    T02 = T01 @ T12
    T03 = T02 @ T23  # base → end-effector

    # Extract positions of each joint / link origin
    # Base (origin)
    p0 = np.array([0.0, 0.0, 0.0])
    # After L1 (shoulder / J2 origin)
    p1 = T01[:3, 3]
    # After L2 (elbow / J3 origin)
    p2 = T02[:3, 3]
    # End-effector
    p3 = T03[:3, 3]

    joint_positions = np.vstack([p0, p1, p2, p3])

    return FKResult(
        joint_positions=joint_positions,
        end_effector_position=p3.copy(),
        end_effector_orientation=T03[:3, :3].copy(),
        transforms=[T01, T02, T03],
        T_ee=T03.copy(),
    )


def compute_joint_positions(
    model: RobotModel, joint_angles: np.ndarray
) -> np.ndarray:
    """Convenience wrapper returning only the (n+1, 3) joint coordinate array."""
    return forward_kinematics(model, joint_angles).joint_positions


def update_state_from_fk(
    model: RobotModel,
    joint_angles: np.ndarray,
    velocities: np.ndarray | None = None,
    accelerations: np.ndarray | None = None,
    timestamp: float = 0.0,
) -> RobotState:
    """Build a RobotState from joint angles via FK."""
    fk = forward_kinematics(model, joint_angles)
    n = model.joint_count
    return RobotState(
        joint_angles=np.asarray(joint_angles, dtype=float),
        joint_velocities=(
            np.asarray(velocities, dtype=float)
            if velocities is not None
            else np.zeros(n)
        ),
        joint_accelerations=(
            np.asarray(accelerations, dtype=float)
            if accelerations is not None
            else np.zeros(n)
        ),
        end_effector_position=fk.end_effector_position,
        end_effector_orientation=fk.end_effector_orientation,
        timestamp=timestamp,
    )
