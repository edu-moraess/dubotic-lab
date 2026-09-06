"""Scene generation tests."""

import numpy as np
from scenes.laboratory import LaboratoryScene
from scenes.roblox import RobloxScene
from scenes.tetris import TetrisScene


def test_laboratory_traces():
    scene = LaboratoryScene()
    traces = scene.get_plotly_traces()
    assert len(traces) > 0
    targets = scene.get_pick_place_targets()
    assert "pick" in targets and "place" in targets
    assert targets["pick"].shape == (3,)


def test_roblox_traces():
    scene = RobloxScene()
    traces = scene.get_plotly_traces()
    assert len(traces) > 0
    targets = scene.get_pick_place_targets()
    assert np.all(np.isfinite(targets["pick"]))


def test_tetris_traces():
    scene = TetrisScene()
    traces = scene.get_plotly_traces()
    assert len(traces) > 0
    targets = scene.get_pick_place_targets()
    assert "pick" in targets
