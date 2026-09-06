"""Hand tracking pipeline tests (no real camera)."""

import numpy as np
import pytest

from vision.smoothing import ExponentialSmoother
from vision.demo import SyntheticHandGenerator
from vision.coordinate_mapping import CoordinateMapper
from vision.gestures import GestureRecognizer, Gesture
from vision.hand_tracking import HandTracker

from robotics.models import RobotModel
from robotics.inverse_kinematics import inverse_kinematics


def test_smoother_reduces_noise():
    sm = ExponentialSmoother(alpha=0.3, dim=3)
    rng = np.random.default_rng(0)
    true = np.array([100.0, 50.0, 40.0])
    outputs = []
    for _ in range(30):
        noisy = true + rng.normal(0, 10, size=3)
        outputs.append(sm.update(noisy))
    final = outputs[-1]
    # Smoothed should be closer to true than a typical noisy sample
    assert np.linalg.norm(final - true) < 8.0


def test_smoother_rejects_nan():
    sm = ExponentialSmoother(alpha=0.5)
    sm.update(np.array([1.0, 2.0, 3.0]))
    out = sm.update(np.array([np.nan, 0.0, 0.0]))
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_synthetic_pipeline_to_ik():
    """hand → mapping → IK without crashing."""
    gen = SyntheticHandGenerator()
    mapper = CoordinateMapper()
    hand = gen.generate(center_norm=(0.6, 0.4), gesture="MOVE")
    tip = hand.index_tip_px
    target = mapper.pixel_to_workspace(tip[0], tip[1])
    assert np.all(np.isfinite(target))

    model = RobotModel.default_3dof()
    ik = inverse_kinematics(model, target)
    # May or may not succeed depending on mapped position; must not raise
    assert ik is not None
    if ik.success:
        assert ik.joint_angles is not None
        assert np.all(np.isfinite(ik.joint_angles))


def test_no_hand_safety():
    """Empty tracker output must not produce NaN targets."""
    tracker = HandTracker()  # may or may not have mediapipe
    # Blank image
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = tracker.process(blank)
    assert not out.has_hand
    assert out.primary is None


def test_gesture_state_machine_inputs():
    gen = SyntheticHandGenerator()
    rec = GestureRecognizer()
    for g in ("MOVE", "GRAB", "RELEASE", "STOP"):
        hand = gen.generate(gesture=g)
        state = rec.recognize(hand.landmarks)
        assert state.gesture != Gesture.UNKNOWN or g == "UNKNOWN"
        assert 0.0 <= state.confidence <= 1.0
