"""
DUBOTIC LAB — Robotics Simulation Laboratory
v0.3 — Real-time vision control + robotics digital twin

Camera → MediaPipe → hand landmarks → fingertip → workspace target → safety → IK → trajectory → robot
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import av
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from robotics.inverse_kinematics import inverse_kinematics, is_reachable
from robotics.kinematics import forward_kinematics
from robotics.models import RobotModel
from robotics.trajectory import TrajectoryPlanner
from scenes.laboratory import LaboratoryScene
from scenes.roblox import RobloxScene
from scenes.tetris import TetrisScene
from vision.coordinate_mapping import CameraBounds, CoordinateMapper, WorkspaceBounds
from vision.gestures import Gesture, GestureRecognizer
from vision.hand_tracking import HandTracker
from vision.smoothing import ExponentialSmoother
from vision.safety import evaluate_motion, target_in_workspace, valid_landmarks
from visualization.robot_plot import create_robot_figure

st.set_page_config(page_title="Dubotic Lab", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {padding-top:1.2rem; padding-bottom:1rem; max-width:1500px;}
.hero {display:flex; justify-content:space-between; align-items:end; margin-bottom:.8rem;}
.hero h1 {font-size:2rem!important; letter-spacing:.08em; margin:0!important;}
.hero p {margin:.15rem 0 0; opacity:.62;}
.section {font-size:.76rem; letter-spacing:.16em; font-weight:700; opacity:.65; margin:.7rem 0 .45rem;}
.panel {border:1px solid rgba(128,128,128,.20); border-radius:14px; padding:12px; background:rgba(20,24,32,.48);}
.pill {display:inline-block; padding:4px 9px; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.05em; border:1px solid rgba(210,215,220,.24); background:rgba(190,195,200,.08);}
.pill-stop {background:rgba(210,210,210,.12); border-color:rgba(245,245,245,.32);}
.small {font-size:.78rem; opacity:.62;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.18); border-radius:12px; padding:.65rem .8rem; background:rgba(20,24,32,.42);}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model() -> RobotModel:
    return RobotModel.default_3dof()


@st.cache_resource
def get_scene_objects():
    return {"Laboratory": LaboratoryScene(), "Roblox-style": RobloxScene(), "Tetris-style": TetrisScene()}


@dataclass
class LiveState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: np.ndarray = field(default_factory=lambda: np.array([150.0, 0.0, 50.0]))
    hand_present: bool = False
    gesture: str = "UNKNOWN"
    confidence: float = 0.0
    fps: float = 0.0
    frames: int = 0
    safety: str = "HOLD — no hand"
    ik_status: str = "HOLD"
    ik_error: float = float("inf")
    status: str = "NO HAND"
    trajectory_points: list = field(default_factory=list)
    last_update: float = 0.0

    def snapshot(self):
        with self.lock:
            return {
                "joint_angles": self.joint_angles.copy(), "target": self.target.copy(),
                "hand_present": self.hand_present, "gesture": self.gesture,
                "confidence": self.confidence, "fps": self.fps, "frames": self.frames,
                "safety": self.safety, "ik_status": self.ik_status,
                "ik_error": self.ik_error, "status": self.status,
                "trajectory_points": list(self.trajectory_points), "last_update": self.last_update,
            }


class LiveHandProcessor:
    """Continuous WebRTC processor. It never writes to Streamlit session_state."""
    def __init__(self, state: LiveState):
        self.state = state
        self.model = RobotModel.default_3dof()
        self.mapper = CoordinateMapper(
            camera=CameraBounds(0, 640, 0, 480),
            workspace=WorkspaceBounds(-180, 180, -120, 120, z=50.0),
            invert_v=True,
        )
        self.tracker = HandTracker(max_hands=1, draw=False)
        self.recognizer = GestureRecognizer()
        self.smoother = ExponentialSmoother(alpha=0.35, dim=3)
        self.planner = TrajectoryPlanner(self.model, default_duration=0.25)
        self.joint_angles = np.zeros(self.model.joint_count)
        self.trajectory = None
        self.traj_index = 0
        self.last_target = None
        self.last_tick = time.perf_counter()
        self.fps_ema = 0.0

    @staticmethod
    def _point(draw, point, radius, outline, fill=None, width=2):
        x, y = float(point[0]), float(point[1])
        box = (x-radius, y-radius, x+radius, y+radius)
        draw.ellipse(box, outline=outline, fill=fill, width=width)

    def _hand_circuit(self, draw, landmarks_px):
        """Monochrome technical HUD: 21 landmarks + anatomical connections."""
        # MediaPipe Hands topology, kept explicit so the overlay does not depend on
        # MediaPipe drawing utilities and remains lightweight in the WebRTC thread.
        connections = (
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        )
        for a, b in connections:
            p1 = landmarks_px[a]
            p2 = landmarks_px[b]
            draw.line((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])), fill=(155,160,165), width=2)

        # Small secondary nodes: restrained gray, like a sensor/kinematic overlay.
        for idx, point in enumerate(landmarks_px):
            if idx == 8:
                continue
            self._point(draw, point, 4, outline=(215,218,222), fill=(55,60,66), width=2)

        # Primary control node: white ring + gray core. No green status color.
        tip = landmarks_px[8]
        self._point(draw, tip, 12, outline=(250,250,250), width=3)
        self._point(draw, tip, 4, outline=(220,220,220), fill=(125,130,135), width=1)

        # Small targeting reticle around the index fingertip.
        x, y = float(tip[0]), float(tip[1])
        draw.line((x-22,y,x-14,y), fill=(225,225,225), width=1)
        draw.line((x+14,y,x+22,y), fill=(225,225,225), width=1)
        draw.line((x,y-22,x,y-14), fill=(225,225,225), width=1)
        draw.line((x,y+14,x,y+22), fill=(225,225,225), width=1)

    def _overlay(self, rgb: np.ndarray, landmarks_px=None, tip=None, label="NO HAND", target=None) -> np.ndarray:
        image = Image.fromarray(rgb.astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        if landmarks_px is not None:
            self._hand_circuit(draw, landmarks_px)
            if target is not None:
                tx = float(np.clip(target[0] / 360.0 * 640.0 + 320.0, 0, 640))
                ty = float(np.clip(240.0 - target[1] / 240.0 * 480.0, 0, 480))
                ix, iy = float(landmarks_px[8,0]), float(landmarks_px[8,1])
                draw.line((ix, iy, tx, ty), fill=(205,208,212), width=1)
                self._point(draw, (tx,ty), 7, outline=(245,245,245), width=2)
                draw.text((tx+10, ty-8), "TARGET", fill=(235,238,242))
        elif tip is not None:
            self._point(draw, tip, 9, outline=(245,245,245), fill=(110,115,120), width=3)

        draw.rounded_rectangle((12, 12, 260, 46), radius=9, fill=(12,16,22), outline=(115,120,125), width=1)
        draw.text((23, 21), label, fill=(235,240,245))
        return np.asarray(image)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        rgb = frame.to_ndarray(format="rgb24")
        h, w = rgb.shape[:2]
        if (w, h) != (640, 480):
            rgb = np.asarray(Image.fromarray(rgb).resize((640, 480)))

        now = time.perf_counter()
        dt = max(now - self.last_tick, 1e-3)
        instant_fps = 1.0 / dt
        self.fps_ema = instant_fps if self.fps_ema == 0 else 0.85*self.fps_ema + 0.15*instant_fps
        self.last_tick = now

        out = self.tracker.process(rgb)
        with self.state.lock:
            self.state.frames += 1
            self.state.fps = self.fps_ema

        if not out.has_hand or out.primary is None:
            self.trajectory = None
            with self.state.lock:
                self.state.hand_present = False
                self.state.status = "NO HAND"
                self.state.gesture = "UNKNOWN"
                self.state.confidence = 0.0
                self.state.safety = "HOLD — no hand"
                self.state.ik_status = "HOLD"
            return av.VideoFrame.from_ndarray(self._overlay(rgb), format="rgb24")

        hand = out.primary
        if not valid_landmarks(hand.landmarks):
            with self.state.lock:
                self.state.hand_present = False
                self.state.status = "INVALID LANDMARKS"
                self.state.safety = "HOLD — invalid landmarks"
            return av.VideoFrame.from_ndarray(self._overlay(rgb, hand.landmarks_px, hand.index_tip_px, "INVALID LANDMARKS"), format="rgb24")

        gesture = self.recognizer.recognize(hand.landmarks)
        with self.state.lock:
            self.state.hand_present = True
            self.state.status = "TRACKING"
            self.state.gesture = gesture.gesture.name
            self.state.confidence = float(gesture.confidence)

        if gesture.gesture == Gesture.STOP:
            self.trajectory = None
            with self.state.lock:
                self.state.safety = "HOLD — STOP"
                self.state.ik_status = "STOP"
            return av.VideoFrame.from_ndarray(self._overlay(rgb, hand.landmarks_px, hand.index_tip_px, "STOP"), format="rgb24")

        try:
            raw = self.mapper.pixel_to_workspace(float(hand.index_tip_px[0]), float(hand.index_tip_px[1]))
            target = self.smoother.update(raw)
        except (TypeError, ValueError):
            with self.state.lock:
                self.state.safety = "HOLD — invalid target"
            return av.VideoFrame.from_ndarray(self._overlay(rgb, hand.landmarks_px, hand.index_tip_px, "INVALID TARGET"), format="rgb24")

        with self.state.lock:
            self.state.target = target.copy()
        if not target_in_workspace(target, self.mapper.workspace):
            self.trajectory = None
            with self.state.lock:
                self.state.safety = "HOLD — outside workspace"
                self.state.ik_status = "OUT OF WORKSPACE"
            return av.VideoFrame.from_ndarray(self._overlay(rgb, hand.landmarks_px, hand.index_tip_px, "WORKSPACE LIMIT", target), format="rgb24")

        if gesture.gesture in (Gesture.MOVE, Gesture.GRAB):
            needs_plan = self.last_target is None or float(np.linalg.norm(target - self.last_target)) > 4.0
            if needs_plan and is_reachable(self.model, target):
                ik = inverse_kinematics(self.model, target)
                decision = evaluate_motion(
                    hand_present=True, landmarks=hand.landmarks, target=target,
                    workspace=self.mapper.workspace,
                    ik_success=bool(ik.success and ik.joint_angles is not None),
                )
                with self.state.lock:
                    self.state.ik_error = float(ik.position_error) if ik.success else float("inf")
                if decision.allow_motion and ik.success and ik.joint_angles is not None:
                    self.trajectory = self.planner.plan(self.joint_angles, ik.joint_angles, n_points=6)
                    self.traj_index = 0
                    self.last_target = target.copy()
                    with self.state.lock:
                        self.state.safety = "CLEAR"
                        self.state.ik_status = "SOLVED"
                else:
                    self.trajectory = None
                    with self.state.lock:
                        self.state.safety = f"HOLD — {decision.reason}"
                        self.state.ik_status = "REJECTED"
            elif not is_reachable(self.model, target):
                self.trajectory = None
                with self.state.lock:
                    self.state.safety = "HOLD — unreachable"
                    self.state.ik_status = "UNREACHABLE"

        if self.trajectory is not None and self.traj_index < self.trajectory.n_points:
            self.joint_angles = self.trajectory.position[self.traj_index].copy()
            self.traj_index += 1

        ee = forward_kinematics(self.model, self.joint_angles).end_effector_position
        with self.state.lock:
            self.state.joint_angles = self.joint_angles.copy()
            self.state.trajectory_points.append(ee.copy())
            self.state.trajectory_points = self.state.trajectory_points[-80:]
            self.state.last_update = now

        return av.VideoFrame.from_ndarray(self._overlay(rgb, hand.landmarks_px, hand.index_tip_px, f"TRACKING · {gesture.gesture.name}", target), format="rgb24")


def get_live_state() -> LiveState:
    if "live_state" not in st.session_state:
        st.session_state.live_state = LiveState()
    return st.session_state.live_state


def reset_live_state():
    st.session_state.live_state = LiveState()


def deg(rad_value):
    return float(np.degrees(rad_value))


def rad(deg_value):
    return float(np.radians(deg_value))


def move_to_target(model, target, current):
    if not is_reachable(model, target):
        return current, None, "Target outside workspace / unreachable."
    ik = inverse_kinematics(model, target)
    if not ik.success or ik.joint_angles is None:
        return current, ik, "IK failed."
    return ik.joint_angles.copy(), ik, f"IK solved · error {ik.position_error:.3f} mm"


def render_live_dashboard(model, scene):
    state = get_live_state()
    snap = state.snapshot()

    st.markdown('<div class="section">VISION CONTROL · REAL-TIME</div>', unsafe_allow_html=True)
    left, right = st.columns([1.08, 1.0], gap="medium")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section">CAMERA / HAND TRACKING</div>', unsafe_allow_html=True)
        webrtc_streamer(
            key="dubotic-live-main",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: LiveHandProcessor(state),
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": {"ideal": 20, "max": 24}}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">White/gray circuit = 21 hand landmarks · fingertip = control node · Z fixed at 50 mm work plane.</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section">DIGITAL TWIN</div>', unsafe_allow_html=True)
        fk = forward_kinematics(model, snap["joint_angles"])
        traj = np.asarray(snap["trajectory_points"]) if snap["trajectory_points"] else None
        fig = create_robot_figure(model, snap["joint_angles"], target=snap["target"], trajectory_points=traj, scene_objects=scene.get_plotly_traces(), title="Live Robot · 3-DOF")
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=35,b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="live_digital_twin")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">LIVE TELEMETRY</div>', unsafe_allow_html=True)
    status_cls = "pill-stop" if snap["gesture"] == "STOP" else ""
    st.markdown(f'<span class="pill {status_cls}">● {snap["status"]}</span> &nbsp; <span class="small">Gesture: {snap["gesture"]}</span>', unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    a.metric("TRACKING", "ACTIVE" if snap["hand_present"] else "LOST")
    b.metric("CONFIDENCE", f'{snap["confidence"]:.2f}')
    c.metric("FPS", f'{snap["fps"]:.1f}')
    d.metric("SAFETY", "CLEAR" if snap["safety"] == "CLEAR" else "HOLD")

    a,b,c,d = st.columns(4)
    a.metric("TARGET X", f'{snap["target"][0]:.1f} mm')
    b.metric("TARGET Y", f'{snap["target"][1]:.1f} mm')
    c.metric("TARGET Z", f'{snap["target"][2]:.1f} mm')
    d.metric("IK", snap["ik_status"])

    a,b,c,d = st.columns(4)
    ee = fk.end_effector_position
    a.metric("EE X", f'{ee[0]:.1f} mm')
    b.metric("EE Y", f'{ee[1]:.1f} mm')
    c.metric("EE Z", f'{ee[2]:.1f} mm')
    d.metric("IK ERROR", "—" if not np.isfinite(snap["ik_error"]) else f'{snap["ik_error"]:.2f} mm')

    st.caption(f'State: {snap["safety"]} · Frames: {snap["frames"]} · Gesture FSM: {snap["gesture"]}')


@st.fragment(run_every="0.5s")
def live_fragment(model, scene):
    render_live_dashboard(model, scene)


def main():
    model = get_model()
    scenes = get_scene_objects()

    st.markdown('<div class="hero"><div><h1>DUBOTIC LAB</h1><p>Robotics Simulation Laboratory · Vision-driven 3-DOF digital twin</p></div></div>', unsafe_allow_html=True)

    top1, top2, top3 = st.columns([1.2,1.2,1])
    with top1:
        scene_name = st.selectbox("SCENE", list(scenes.keys()), key="scene_select")
    with top2:
        mode = st.selectbox("CONTROL MODE", ["Manual Control", "Target Position", "Hand Tracking"], key="mode_select")
    with top3:
        task = st.selectbox("TASK", ["Free Move", "Reach Target", "Pick & Place"], key="task_select")
    scene = scenes[scene_name]

    if mode == "Hand Tracking":
        live_fragment(model, scene)
        return

    st.markdown('<div class="section">ROBOT DIGITAL TWIN</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1.0], gap="medium")
    with left:
        fig = create_robot_figure(model, st.session_state.get("manual_angles", np.zeros(3)), target=None if mode == "Manual Control" else np.array([150.,0.,50.]), scene_objects=scene.get_plotly_traces(), title=f"{scene.name} · 3-DOF Arm")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if mode == "Manual Control":
            st.markdown('<div class="section">JOINT CONTROL</div>', unsafe_allow_html=True)
            angles = st.session_state.get("manual_angles", np.zeros(3))
            limits = [(np.degrees(a), np.degrees(b)) for a,b in model.joint_limits]
            vals=[]
            for i,((lo,hi),val) in enumerate(zip(limits, angles),1):
                vals.append(rad(st.slider(f"J{i} · degrees", float(lo), float(hi), float(deg(val)), 1.0, key=f"manual_j{i}")))
            st.session_state.manual_angles=np.array(vals)
            fk=forward_kinematics(model, st.session_state.manual_angles)
            st.metric("END EFFECTOR", "%.1f, %.1f, %.1f mm" % tuple(fk.end_effector_position))
        else:
            st.markdown('<div class="section">TARGET CONTROL</div>', unsafe_allow_html=True)
            target = st.session_state.get("target_manual", np.array([150.,0.,50.]))
            x,y,z=[st.number_input(label, value=float(value), step=5.0, key=f"target_{axis}") for label,value,axis in [("X (mm)",target[0],"x"),("Y (mm)",target[1],"y"),("Z (mm)",target[2],"z")]]
            target=np.array([x,y,z]); st.session_state.target_manual=target
            if st.button("MOVE TO TARGET", type="primary", use_container_width=True):
                angles, ik, msg=move_to_target(model,target,st.session_state.get("manual_angles",np.zeros(3)))
                if ik is not None: st.session_state.last_target_ik=ik
                if ik is not None and ik.success: st.session_state.manual_angles=angles
                st.session_state.target_message=msg
            if st.session_state.get("target_message"): st.info(st.session_state.target_message)
            if st.session_state.get("last_target_ik") is not None:
                ik=st.session_state.last_target_ik
                st.metric("IK ERROR", f"{ik.position_error:.3f} mm")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
