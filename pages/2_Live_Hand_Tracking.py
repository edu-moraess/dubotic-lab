"""Dubotic Lab v0.3.1 — lightweight real-time browser hand tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

from robotics.inverse_kinematics import inverse_kinematics, is_reachable
from robotics.models import RobotModel
from robotics.trajectory import TrajectoryPlanner
from visualization.robot_plot import create_robot_figure
from vision.coordinate_mapping import CameraBounds, CoordinateMapper, WorkspaceBounds
from vision.gestures import Gesture, GestureRecognizer
from vision.hand_tracking import HandTracker
from vision.safety import evaluate_motion, target_in_workspace, valid_landmarks
from vision.smoothing import ExponentialSmoother

st.set_page_config(page_title="Dubotic Lab — Live Hand Tracking", page_icon="🤖", layout="centered")

@dataclass
class LiveState:
    lock: threading.Lock
    joint_angles: np.ndarray
    target: np.ndarray
    gesture: str = "UNKNOWN"
    confidence: float = 0.0
    status: str = "WAITING"
    safety: str = "WAITING"
    ik_ok: bool = False
    ik_error: float = float("inf")
    frames: int = 0
    fps: float = 0.0
    last_time: float = 0.0

class LiveHandProcessor(VideoProcessorBase):
    def __init__(self, state: LiveState):
        self.state = state
        self.model = RobotModel.default_3dof()
        self.mapper = CoordinateMapper(CameraBounds(0, 640, 0, 480), WorkspaceBounds(-180, 180, -120, 120, z=50.0), invert_v=True)
        self.smoother = ExponentialSmoother(alpha=0.35, dim=3)
        self.tracker = HandTracker(max_hands=1, draw=False)
        self.recognizer = GestureRecognizer()
        self.planner = TrajectoryPlanner(self.model, default_duration=0.15)
        self.trajectory = None
        self.traj_index = 0
        self.last_target = None

    def _set(self, **values):
        with self.state.lock:
            for key, value in values.items():
                setattr(self.state, key, value)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        rgb = frame.to_ndarray(format="rgb24")
        h, w = rgb.shape[:2]
        out = self.tracker.process(rgb)
        now = time.monotonic()
        with self.state.lock:
            self.state.frames += 1
            previous = self.state.last_time
            self.state.last_time = now
            if previous > 0:
                dt = now - previous
                if dt > 0:
                    instant = 1.0 / dt
                    self.state.fps = instant if self.state.fps == 0 else 0.9 * self.state.fps + 0.1 * instant
        annotated = rgb.copy()
        if not out.has_hand:
            self._set(status="NO HAND", gesture="UNKNOWN", confidence=0.0, safety="HOLD — no hand", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        hand = out.primary
        if hand is None or not valid_landmarks(hand.landmarks):
            self._set(status="INVALID LANDMARKS", safety="HOLD — invalid landmarks", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        gesture = self.recognizer.recognize(hand.landmarks)
        self._set(status="ACTIVE", gesture=gesture.gesture.name, confidence=float(gesture.confidence))
        px = hand.index_tip_px
        if gesture.gesture == Gesture.STOP:
            self.trajectory = None
            self.last_target = None
            self._set(status="STOP", safety="HOLD — STOP gesture", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        try:
            px_x = float(px[0]) * 640.0 / max(w, 1) if w != 640 else float(px[0])
            px_y = float(px[1]) * 480.0 / max(h, 1) if h != 480 else float(px[1])
            target = self.smoother.update(self.mapper.pixel_to_workspace(px_x, px_y))
        except (TypeError, ValueError):
            self._set(safety="HOLD — invalid target", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        if not target_in_workspace(target, self.mapper.workspace):
            self._set(target=target.copy(), safety="HOLD — target outside workspace", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        if gesture.gesture not in (Gesture.MOVE, Gesture.GRAB):
            self._set(target=target.copy(), safety="HOLD — gesture does not command motion", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        ik = inverse_kinematics(self.model, target) if is_reachable(self.model, target) else None
        decision = evaluate_motion(hand_present=True, landmarks=hand.landmarks, target=target, workspace=self.mapper.workspace, ik_success=bool(ik is not None and ik.success and ik.joint_angles is not None))
        if not decision.allow_motion or ik is None or ik.joint_angles is None:
            self._set(target=target.copy(), safety=f"HOLD — {decision.reason}", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")
        if self.last_target is None or np.linalg.norm(target - self.last_target) > 4.0:
            with self.state.lock:
                current = self.state.joint_angles.copy()
            self.trajectory = self.planner.plan(current, ik.joint_angles, n_points=6)
            self.traj_index = 0
            self.last_target = target.copy()
        next_angles = ik.joint_angles.copy()
        if self.trajectory is not None and self.traj_index < self.trajectory.n_points:
            next_angles = self.trajectory.position[self.traj_index].copy()
            self.traj_index += 1
        self._set(joint_angles=next_angles, target=target.copy(), ik_ok=True, ik_error=float(ik.position_error), safety="VALID")
        return av.VideoFrame.from_ndarray(annotated, format="rgb24")

@st.cache_resource
def get_model() -> RobotModel:
    return RobotModel.default_3dof()

def render_live_ui(state: LiveState, model: RobotModel):
    with state.lock:
        target, joints = state.target.copy(), state.joint_angles.copy()
        gesture, confidence, status, safety = state.gesture, state.confidence, state.status, state.safety
        ik_ok, ik_error, fps = state.ik_ok, state.ik_error, state.fps
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Gesture", gesture)
    c3.metric("Confidence", f"{confidence:.2f}")
    c4.metric("FPS", f"{fps:.1f}")
    st.write(f"**Target:** X={target[0]:.1f} · Y={target[1]:.1f} · Z={target[2]:.1f} mm · **IK:** {'VALID' if ik_ok else 'HOLD'} · erro={ik_error:.2f} mm")
    st.write(f"**Safety:** {safety}")
    # Plotly is intentionally refreshed slowly; video processing stays in WebRTC worker.
    fig = create_robot_figure(model, joints, target=target, title="Live Hand Control · 3-DOF Arm")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def main():
    st.title("DUBOTIC LAB")
    st.caption("v0.3.1 · Real-Time Hand Tracking")
    st.info("Câmera contínua: dedo indicador → target → IK → braço virtual. Z permanece fixo em 50 mm.")
    model = get_model()
    if "live_state" not in st.session_state:
        st.session_state.live_state = LiveState(threading.Lock(), np.zeros(model.joint_count), np.array([150.0, 0.0, 50.0]))
    state = st.session_state.live_state
    ctx = webrtc_streamer(key="dubotic-live-hand", mode=WebRtcMode.SENDRECV, video_processor_factory=lambda: LiveHandProcessor(state), media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 20}, "audio": False}, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, async_processing=True)
    if ctx.state.playing:
        st.success("LIVE — câmera conectada")
        @st.fragment(run_every="0.5s")
        def live_panel():
            render_live_ui(state, model)
        live_panel()
    else:
        st.warning("Press START acima para abrir a câmera. Em celular, permita acesso à câmera quando solicitado.")
    st.caption("Pipeline: câmera → MediaPipe tracking → index tip → EMA → target → IK → trajetória → robô")

if __name__ == "__main__":
    main()
