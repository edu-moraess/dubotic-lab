"""Inverse kinematics + verification tests."""

import numpy as np
import pytest

from robotics.models import RobotModel
from robotics.kinematics import forward_kinematics
from robotics.inverse_kinematics import inverse_kinematics, is_reachable, DEFAULT_TOLERANCE


@pytest.fixture
def model():
    return RobotModel.default_3dof()


def test_ik_fk_roundtrip(model):
    """angles → FK → target → IK → FK  must recover position within tolerance."""
    rng = np.random.default_rng(123)
    for _ in range(20):
        # Sample inside limits
        q = np.array(
            [
                rng.uniform(*model.joint_limits[0]),
                rng.uniform(*model.joint_limits[1]),
                rng.uniform(*model.joint_limits[2]),
            ]
        )
        fk = forward_kinematics(model, q)
        target = fk.end_effector_position

        ik = inverse_kinematics(model, target, tolerance=DEFAULT_TOLERANCE)
        assert ik.success, ik.message
        assert ik.position_error < DEFAULT_TOLERANCE

        # Re-verify
        fk2 = forward_kinematics(model, ik.joint_angles)
        err = np.linalg.norm(fk2.end_effector_position - target)
        assert err < DEFAULT_TOLERANCE


def test_unreachable(model):
    far = np.array([1000.0, 1000.0, 1000.0])
    assert not is_reachable(model, far)
    ik = inverse_kinematics(model, far)
    assert not ik.success
    assert "unreachable" in ik.message.lower()


def test_home_ik(model):
    L1, L2, L3 = model.link_lengths
    target = np.array([L2 + L3, 0.0, L1])
    ik = inverse_kinematics(model, target)
    assert ik.success
    np.testing.assert_allclose(ik.joint_angles, [0, 0, 0], atol=1e-5)
