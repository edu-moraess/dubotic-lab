"""Unit tests for homogeneous transforms."""

import numpy as np
import pytest

from robotics.transforms import (
    rotation_x,
    rotation_y,
    rotation_z,
    translation,
    homogeneous_transform,
    validate_rotation,
    compose,
)


def test_identity_rotation():
    for R in (rotation_x(0), rotation_y(0), rotation_z(0)):
        valid, ortho, det = validate_rotation(R)
        assert valid
        assert ortho < 1e-10
        assert det < 1e-10
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_rotation_orthogonality():
    angles = [0.1, 0.7, 1.5, np.pi / 2, -0.3]
    for th in angles:
        for R in (rotation_x(th), rotation_y(th), rotation_z(th)):
            valid, ortho, det = validate_rotation(R)
            assert valid, f"ortho={ortho}, det={det}"
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_translation():
    T = translation(10, -5, 3)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], np.eye(3))
    np.testing.assert_allclose(T[:3, 3], [10, -5, 3])
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])


def test_homogeneous_transform():
    R = rotation_z(np.pi / 4)
    t = np.array([1.0, 2.0, 3.0])
    T = homogeneous_transform(R, t)
    np.testing.assert_allclose(T[:3, :3], R)
    np.testing.assert_allclose(T[:3, 3], t)


def test_compose():
    T1 = translation(1, 0, 0)
    T2 = translation(0, 2, 0)
    T = compose(T1, T2)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 0])
