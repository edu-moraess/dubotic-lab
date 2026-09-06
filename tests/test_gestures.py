"""Gesture recognition tests with synthetic landmarks."""

import numpy as np
import pytest

from vision.gestures import Gesture, GestureRecognizer
from vision.demo import SyntheticHandGenerator


@pytest.fixture
def recognizer():
    return GestureRecognizer()


@pytest.fixture
def gen():
    return SyntheticHandGenerator(image_size=(640, 480))


def test_move(recognizer, gen):
    hand = gen.generate(gesture="MOVE")
    state = recognizer.recognize(hand.landmarks)
    assert state.gesture == Gesture.MOVE


def test_grab(recognizer, gen):
    hand = gen.generate(gesture="GRAB")
    state = recognizer.recognize(hand.landmarks)
    assert state.gesture == Gesture.GRAB


def test_release(recognizer, gen):
    hand = gen.generate(gesture="RELEASE")
    state = recognizer.recognize(hand.landmarks)
    assert state.gesture == Gesture.RELEASE


def test_stop(recognizer, gen):
    hand = gen.generate(gesture="STOP")
    state = recognizer.recognize(hand.landmarks)
    assert state.gesture == Gesture.STOP


def test_unknown_shape(recognizer):
    state = recognizer.recognize(np.zeros((10, 2)))
    assert state.gesture == Gesture.UNKNOWN
