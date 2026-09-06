"""Coordinate mapping unit tests."""

import numpy as np
import pytest

from vision.coordinate_mapping import (
    CoordinateMapper,
    CameraBounds,
    WorkspaceBounds,
)


@pytest.fixture
def mapper():
    return CoordinateMapper(
        camera=CameraBounds(0, 640, 0, 480),
        workspace=WorkspaceBounds(-200, 200, -150, 150, z=50.0),
        invert_v=True,
    )


def test_center(mapper):
    xyz = mapper.pixel_to_workspace(320, 240)
    np.testing.assert_allclose(xyz[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(xyz[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(xyz[2], 50.0)


def test_corners(mapper):
    # top-left pixel → depends on invert_v
    xyz = mapper.pixel_to_workspace(0, 0)
    assert xyz[0] == pytest.approx(-200.0)
    assert xyz[1] == pytest.approx(150.0)  # inverted
    assert xyz[2] == 50.0

    xyz = mapper.pixel_to_workspace(640, 480)
    assert xyz[0] == pytest.approx(200.0)
    assert xyz[1] == pytest.approx(-150.0)


def test_clamp(mapper):
    xyz = mapper.pixel_to_workspace(-100, 1000)
    assert xyz[0] == pytest.approx(-200.0)
    assert xyz[1] == pytest.approx(-150.0)


def test_invalid():
    m = CoordinateMapper()
    with pytest.raises(ValueError):
        m.pixel_to_workspace(np.nan, 10)


def test_roundtrip(mapper):
    u, v = 100.0, 200.0
    xyz = mapper.pixel_to_workspace(u, v)
    u2, v2 = mapper.workspace_to_pixel(xyz[0], xyz[1])
    np.testing.assert_allclose([u2, v2], [u, v], atol=1e-6)
