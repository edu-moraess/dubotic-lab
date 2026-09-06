"""PID controller tests."""

import numpy as np
import pytest

from robotics.controller import PIDController


def test_proportional_response():
    pid = PIDController.scalar(kp=2.0, ki=0.0, kd=0.0)
    u = pid.update(target=np.array([1.0]), measurement=np.array([0.0]), dt=0.1)
    np.testing.assert_allclose(u, [2.0])


def test_integral_accumulates():
    pid = PIDController.scalar(kp=0.0, ki=1.0, kd=0.0)
    pid.update(np.array([1.0]), np.array([0.0]), dt=0.5)
    u2 = pid.update(np.array([1.0]), np.array([0.0]), dt=0.5)
    # integral ≈ 1.0 after two steps of 0.5
    np.testing.assert_allclose(u2, [1.0], atol=1e-9)


def test_output_saturation():
    pid = PIDController.scalar(kp=10.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0))
    u = pid.update(np.array([1.0]), np.array([0.0]), dt=0.1)
    np.testing.assert_allclose(u, [1.0])


def test_reset():
    pid = PIDController.scalar(1.0, 1.0, 0.0)
    pid.update(np.array([1.0]), np.array([0.0]), dt=0.1)
    pid.reset()
    assert np.allclose(pid._integral, 0.0)
