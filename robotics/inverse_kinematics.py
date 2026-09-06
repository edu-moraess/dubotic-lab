"""
Inverse Kinematics for the 3-DOF Dubotic arm.

Geometry (matches corrected FK – Rot @ Trans convention):

  After base yaw θ1:
    r  = sqrt(x² + y²)
    zs = z - L1

  Two-link planar arm in the (r, zs) plane with rotations about Y:

    r  =  L2·cos(θ2) + L3·cos(θ2+θ3)
    zs = -L2·sin(θ2) - L3·sin(θ2+θ3)

Verification is mandatory: IK → FK → error check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .models import RobotModel
from .kinematics import forward_kinematics


DEFAULT_TOLERANCE = 1.0  # mm


@dataclass
class IKResult:
    success: bool
    joint_angles: Optional[np.ndarray]
    position_error: float
    message: str
    reachable: bool


def is_reachable(
    model: RobotModel,
    target: np.ndarray,
    margin: float = 0.5,
) -> bool:
    """Conservative geometric reachability (ignores joint limits)."""
    target = np.asarray(target, dtype=float).ravel()
    L1, L2, L3 = model.link_lengths
    x, y, z = target
    r = np.sqrt(x**2 + y**2)
    zs = z - L1
    d = np.sqrt(r**2 + zs**2)
    max_d = L2 + L3 + 1e-6
    min_d = abs(L2 - L3) + margin * 0.2
    return min_d <= d <= max_d


def inverse_kinematics(
    model: RobotModel,
    target: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    prefer_elbow_up: bool = True,
) -> IKResult:
    """
    Analytic inverse kinematics consistent with the FK transform chain.
    """
    target = np.asarray(target, dtype=float).ravel()
    if target.shape[0] != 3:
        return IKResult(
            False, None, np.inf, "Target must be a 3-vector (x, y, z)", False
        )

    if not is_reachable(model, target):
        return IKResult(
            False, None, np.inf,
            "Target unreachable (outside geometric workspace)", False
        )

    L1, L2, L3 = model.link_lengths
    x, y, z = target

    theta1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    zs = z - L1
    d2 = r**2 + zs**2

    # Cosine law for θ3
    cos_th3 = (d2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_th3 = np.clip(cos_th3, -1.0, 1.0)
    sin_th3 = np.sqrt(max(0.0, 1.0 - cos_th3**2))

    candidates = []

    signs = (-1, 1) if prefer_elbow_up else (1, -1)
    for sign in signs:
        th3 = sign * np.arctan2(sin_th3, cos_th3)

        k1 = L2 + L3 * np.cos(th3)
        k2 = L3 * np.sin(th3)
        th2 = np.arctan2(-zs, r) - np.arctan2(k2, k1)

        q = np.array([theta1, th2, th3], dtype=float)
        q = model.clamp_joints(q)
        if model.within_limits(q):
            candidates.append(q)

    if not candidates:
        return IKResult(
            False, None, np.inf, "No solution within joint limits", True
        )

    best = None
    best_err = np.inf
    for q in candidates:
        fk = forward_kinematics(model, q)
        err = float(np.linalg.norm(fk.end_effector_position - target))
        if err < best_err:
            best_err = err
            best = q

    if best is None or best_err > tolerance:
        return IKResult(
            False,
            best,
            best_err if best is not None else np.inf,
            f"IK solution failed verification (error={best_err:.3f} mm > {tolerance} mm)",
            True,
        )

    return IKResult(True, best, best_err, "IK succeeded", True)
