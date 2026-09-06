"""Trajectory planner tests."""

import numpy as np
import pytest

from robotics.models import RobotModel
from robotics.trajectory import TrajectoryPlanner


@pytest.fixture
def model():
    return RobotModel.default_3dof()


def test_start_end(model):
    planner = TrajectoryPlanner(model, default_duration=1.5)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.5, -0.3, 0.4])
    traj = planner.plan(start, goal, n_points=50)

    np.testing.assert_allclose(traj.position[0], start, atol=1e-12)
    np.testing.assert_allclose(traj.position[-1], goal, atol=1e-12)
    assert traj.duration == 1.5
    assert traj.n_points == 50


def test_zero_velocity_boundaries(model):
    planner = TrajectoryPlanner(model)
    start = np.zeros(3)
    goal = np.array([1.0, 0.5, -0.5])
    traj = planner.plan(start, goal, duration=2.0, n_points=100)
    np.testing.assert_allclose(traj.velocity[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(traj.velocity[-1], 0.0, atol=1e-10)


def test_path_length_positive(model):
    planner = TrajectoryPlanner(model)
    traj = planner.plan(np.zeros(3), np.array([0.8, 0.4, -0.3]))
    assert traj.path_length > 0
    assert traj.max_velocity > 0
