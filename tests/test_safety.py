"""Safety gates for hand tracking; no camera is used."""
import numpy as np

from vision.coordinate_mapping import WorkspaceBounds
from vision.safety import evaluate_motion, target_in_workspace, valid_landmarks


def _landmarks():
    return np.zeros((21, 3), dtype=float)


def test_valid_landmarks_and_invalid_shapes():
    assert valid_landmarks(_landmarks())
    assert not valid_landmarks(np.zeros((20, 3)))
    bad = _landmarks()
    bad[8, 0] = np.nan
    assert not valid_landmarks(bad)


def test_workspace_limits_and_fixed_z():
    bounds = WorkspaceBounds(-10, 10, -20, 20, z=50)
    assert target_in_workspace(np.array([0, 0, 50]), bounds)
    assert not target_in_workspace(np.array([11, 0, 50]), bounds)
    assert not target_in_workspace(np.array([0, 0, 51]), bounds)


def test_stop_has_priority():
    decision = evaluate_motion(
        hand_present=True,
        landmarks=_landmarks(),
        target=np.array([0, 0, 50]),
        workspace=WorkspaceBounds(-10, 10, -20, 20, z=50),
        ik_success=True,
        stop_gesture=True,
    )
    assert not decision.allow_motion
    assert decision.reason == "STOP gesture"


def test_missing_hand_invalid_target_and_ik_hold():
    bounds = WorkspaceBounds(-10, 10, -20, 20, z=50)
    for kwargs in (
        dict(hand_present=False, landmarks=_landmarks(), target=np.array([0, 0, 50]), ik_success=True),
        dict(hand_present=True, landmarks=_landmarks(), target=np.array([99, 0, 50]), ik_success=True),
        dict(hand_present=True, landmarks=_landmarks(), target=np.array([0, 0, 50]), ik_success=False),
    ):
        decision = evaluate_motion(workspace=bounds, stop_gesture=False, **kwargs)
        assert not decision.allow_motion
