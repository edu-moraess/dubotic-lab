"""
Homogeneous transformation utilities (4x4 matrices).

All rotations are pure orthogonal matrices with det ≈ +1.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def rotation_x(theta: float) -> np.ndarray:
    """Rotation matrix about X axis (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )
    return R


def rotation_y(theta: float) -> np.ndarray:
    """Rotation matrix about Y axis (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )
    return R


def rotation_z(theta: float) -> np.ndarray:
    """Rotation matrix about Z axis (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return R


def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    """4x4 pure translation homogeneous matrix."""
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous transform from 3x3 rotation and 3-vector translation.
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def validate_rotation(
    R: np.ndarray, tol: float = 1e-6
) -> Tuple[bool, float, float]:
    """
    Validate that R is a proper rotation matrix.

    Returns
    -------
    is_valid : bool
    ortho_error : float   ||R @ R.T - I||_F
    det_error : float     |det(R) - 1|
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False, np.inf, np.inf
    ortho_err = np.linalg.norm(R @ R.T - np.eye(3), ord="fro")
    det_err = abs(np.linalg.det(R) - 1.0)
    valid = ortho_err < tol and det_err < tol
    return valid, float(ortho_err), float(det_err)


def compose(*Ts: np.ndarray) -> np.ndarray:
    """Left-to-right composition of homogeneous transforms: T0 @ T1 @ ..."""
    result = np.eye(4, dtype=float)
    for T in Ts:
        result = result @ T
    return result
