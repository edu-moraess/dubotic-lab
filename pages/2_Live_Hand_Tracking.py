"""Dubotic Lab v0.3 — real-time browser hand tracking.

Uses streamlit-webrtc for a continuous camera stream. The existing robotics
core is reused: hand -> target -> IK -> trajectory -> robot state.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
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
        self.mapper = CoordinateMapper(
            camera=CameraBounds(0, 640, 0, 480),
            workspace=WorkspaceBounds(-180, 180, -120, 120, z=50.0),
            invert_v=True,
        )
        self.smoother = ExponentialSmoother(alpha=0.35, dim=3)
        self.tracker = HandTracker(max_hands=1, draw=True)
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
        if (w, h) != (640, 480):
            rgb = np.asarray(Image.fromarray(rgb).resize((640, 480)))

        out = self.tracker.process(rgb)
        with self.state.lock:
            self.state.frames += 1
            now = time.monotonic()
            previous = self.state.last_time
            self.state.last_time = now
            if previous > 0:
                dt = now - previous
                if dt > 0:
                    self.state.fps = 0.9 * self.state.fps + 0.1 * (1.0 / dt)

        annotated = out.annotated_image if out.annotated_image is not None else rgb.copy()
        if not out.has_hand:
            self._set(status="NO HAND", gesture="UNKNOWN", confidence=0.0, safety="HOLD — no hand", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")

        hand = out.primary
        if hand is None or not valid_landmarks(hand.landmarks):
            self._set(status="INVALID LANDMARKS", safety="HOLD — invalid landmarks", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")

        gesture = self.recognizer.recognize(hand.landmarks)
        self._set(status="ACTIVE", gesture=gesture.gesture.name, confidence=float(gesture.confidence))

        overlay = Image.fromarray(annotated).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        px = hand.index_tip_px
        r = 9
        draw.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r), fill=(255, 60, 60), outline=(255, 255, 255), width=2)
        annotated = np.asarray(overlay)

        if gesture.gesture == Gesture.STOP:
            self.trajectory = None
            self._set(status="STOP", safety="HOLD — STOP gesture", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")

        try:
            raw = self.mapper.pixel_to_workspace(float(px[0]), float(px[1]))
            target = self.smoother.update(raw)
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
        decision = evaluate_motion(
            hand_present=True,
            landmarks=hand.landmarks,
            target=target,
            workspace=self.mapper.workspace,
            ik_success=bool(ik is not None and ik.success and ik.joint_angles is not None),
        )
        if not decision.allow_motion or ik is None or ik.joint_angles is None:
            self._set(target=target.copy(), safety=f"HOLD — {decision.reason}", ik_ok=False)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")

        if self.last_target is None or np.linalg.norm(target - self.last_target) > 1.0:
            with self.state.lock:
                current = self.state.joint_angles.copy()
            self.trajectory = self.planner.plan(current, ik.joint_angles, n_points=8)
            self.traj_index = 0
            self.last_target = target.copy()

        if self.trajectory is not None and self.traj_index < self.trajectory.n_points:
            next_angles = self.trajectory.position[self.traj_index].copy()
            self.traj_index += 1
        else:
            next_angles = ik.joint_angles.copy()

        self._set(
            joint_angles=next_angles,
            target=target.copy(),
            ik_ok=True,
            ik_error=float(ik.position_error),
            safety=f"VALID — {self.trajectory.n_points if self.trajectory is not None else 0} point trajectory",
        )
        return av.VideoFrame.from_ndarray(annotated, format="rgb24")


@st.cache_resource
def get_model() -> RobotModel:
    return RobotModel.default_3dof()


def main():
    st.title("DUBOTIC LAB")
    st.caption("v0.3 · Real-Time Hand Tracking")
    st.info("Câmera contínua: mova o dedo indicador e o braço virtual acompanha. Z permanece fixo no plano de trabalho.")

    model = get_model()
    if "live_state" not in st.session_state:
        st.session_state.live_state = LiveState(
            lock=threading.Lock(),
            joint_angles=np.zeros(model.joint_count),
            target=np.array([150.0, 0.0, 50.0]),
        )

    state = st.session_state.live_state
    ctx = webrtc_streamer(
        key="dubotic-live-hand",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: LiveHandProcessor(state),
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    if ctx.state.playing:
        st.success("LIVE — câmera conectada")
        telemetry = st.empty()
        robot_box = st.empty()
        while ctx.state.playing:
            with state.lock:
                target = state.target.copy()
                joints = state.joint_angles.copy()
                gesture = state.gesture
                confidence = state.confidence
                status = state.status
                safety = state.safety
                ik_ok = state.ik_ok
                ik_error = state.ik_error
                fps = state.fps

            telemetry.markdown(
                f"**{status}** · gesto **{gesture}** · confiança **{confidence:.2f}** · FPS **{fps:.1f}**\n\n"
                f"**Target:** X={target[0]:.1f} · Y={target[1]:.1f} · Z={target[2]:.1f} mm  · "
                f"**IK:** {'VALID' if ik_ok else 'HOLD'} · erro={ik_error:.2f} mm  · **Safety:** {safety}"
            )
            fig = create_robot_figure(model, joints, target=target, title="Live Hand Control · 3-DOF Arm")
            robot_box.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key="live_robot_plot",
            )
            time.sleep(0.10)
    else:
        st.warning("Press START acima para abrir a câmera. Em celular, permita acesso à câmera quando solicitado.")

    st.caption("Arquitetura: câmera → MediaPipe → index tip → EMA → target → IK → trajetória → robô")


if __name__ == "__main__":
    main()
