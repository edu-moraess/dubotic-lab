"""Forward kinematics tests."""

import numpy as np
import pytest

from robotics.models import RobotModel
from robotics.kinematics import forward_kinematics, compute_joint_positions


@pytest.fixture
def model():
    return RobotModel.default_3dof()


def test_home_position(model):
    """All joints zero → EE should be at (L2+L3, 0, L1)."""
    L1, L2, L3 = model.link_lengths
    fk = forward_kinematics(model, np.zeros(3))
    expected = np.array([L2 + L3, 0.0, L1])
    np.testing.assert_allclose(fk.end_effector_position, expected, atol=1e-9)
    assert fk.joint_positions.shape == (4, 3)


def test_joint_positions_chain(model):
    pts = compute_joint_positions(model, np.zeros(3))
    L1, L2, L3 = model.link_lengths
    np.testing.assert_allclose(pts[0], [0, 0, 0])
    np.testing.assert_allclose(pts[1], [0, 0, L1], atol=1e-9)
    np.testing.assert_allclose(pts[2], [L2, 0, L1], atol=1e-9)
    np.testing.assert_allclose(pts[3], [L2 + L3, 0, L1], atol=1e-9)


def test_base_rotation(model):
    """θ1 = 90° should rotate the arm into the YZ plane."""
    L1, L2, L3 = model.link_lengths
    q = np.array([np.pi / 2, 0.0, 0.0])
    fk = forward_kinematics(model, q)
    expected = np.array([0.0, L2 + L3, L1])
    np.testing.assert_allclose(fk.end_effector_position, expected, atol=1e-8)


def test_orientation_is_rotation(model):
    from robotics.transforms import validate_rotation

    for q in [
        np.zeros(3),
        np.array([0.3, -0.4, 0.5]),
        np.array([1.0, 0.5, -0.8]),
    ]:
        fk = forward_kinematics(model, q)
        valid, _, _ = validate_rotation(fk.end_effector_orientation)
        assert valid
