"""
DUBOTIC LAB – Interactive Robotics Simulation Laboratory
Mobile-first Streamlit front-end.
"""

from __future__ import annotations

import streamlit as st
import numpy as np

from robotics.models import RobotModel, RobotState
from robotics.kinematics import forward_kinematics, update_state_from_fk
from robotics.inverse_kinematics import inverse_kinematics, is_reachable
from robotics.trajectory import TrajectoryPlanner
from robotics.controller import PIDController
from robotics.simulation import Simulation, JointPositionSensor

from visualization.robot_plot import create_robot_figure
from visualization.workspace_plot import sample_workspace, create_workspace_figure
from visualization.trajectory_plot import create_trajectory_plots

from scenes.laboratory import LaboratoryScene
from scenes.roblox import RobloxScene
from scenes.tetris import TetrisScene


# ---------------------------------------------------------------------------
# Page config – mobile friendly
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dubotic Lab",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for touch-friendly controls
st.markdown(
    """
    <style>
    .stSlider > div { padding-top: 0.4rem; padding-bottom: 0.4rem; }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.05rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1a1f2e;
        padding: 0.6rem;
        border-radius: 0.4rem;
    }
    h1 { font-size: 1.6rem !important; margin-bottom: 0.2rem; }
    h2, h3 { font-size: 1.15rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model() -> RobotModel:
    return RobotModel.default_3dof()


@st.cache_data(show_spinner=False)
def get_workspace_points(_model: RobotModel) -> np.ndarray:
    return sample_workspace(_model, n_samples_per_joint=10)


SCENES = {
    "Laboratory": LaboratoryScene(),
    "Roblox-style": RobloxScene(),
    "Tetris-style": TetrisScene(),
}


def deg(rad: float) -> float:
    return float(np.degrees(rad))


def rad(deg_val: float) -> float:
    return float(np.radians(deg_val))


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state(model: RobotModel):
    if "joint_angles" not in st.session_state:
        st.session_state.joint_angles = np.zeros(model.joint_count)
    if "target" not in st.session_state:
        st.session_state.target = np.array([150.0, 0.0, 150.0])
    if "last_ik" not in st.session_state:
        st.session_state.last_ik = None
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = None
    if "traj_history" not in st.session_state:
        st.session_state.traj_history = None
    if "mode" not in st.session_state:
        st.session_state.mode = "Manual Control"
    if "scene_name" not in st.session_state:
        st.session_state.scene_name = "Laboratory"
    if "task" not in st.session_state:
        st.session_state.task = "Free Move"
    if "noise_on" not in st.session_state:
        st.session_state.noise_on = False
    if "status_msg" not in st.session_state:
        st.session_state.status_msg = ""
    if "status_type" not in st.session_state:
        st.session_state.status_type = "info"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model = get_model()
    init_state(model)

    st.title("DUBOTIC LAB")
    st.caption("«Robotics Simulation Laboratory»")

    # ---- Scene selector ----
    scene_name = st.selectbox(
        "SCENE",
        list(SCENES.keys()),
        index=list(SCENES.keys()).index(st.session_state.scene_name),
    )
    st.session_state.scene_name = scene_name
    scene = SCENES[scene_name]

    # ---- Mode / Task ----
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mode = st.selectbox(
            "MODE",
            ["Manual Control", "Target Position"],
            index=0 if st.session_state.mode == "Manual Control" else 1,
        )
        st.session_state.mode = mode
    with col_m2:
        task = st.selectbox(
            "TASK",
            ["Free Move", "Reach Target", "Pick & Place"],
            index=["Free Move", "Reach Target", "Pick & Place"].index(
                st.session_state.task
            ),
        )
        st.session_state.task = task

    # ---- 3D View ----
    fk = forward_kinematics(model, st.session_state.joint_angles)
    traj_pts = None
    if st.session_state.traj_history is not None:
        traj_pts = np.array(
            [s.end_effector_position for s in st.session_state.traj_history]
        )

    scene_traces = scene.get_plotly_traces()
    fig = create_robot_figure(
        model,
        st.session_state.joint_angles,
        target=st.session_state.target if mode == "Target Position" or task != "Free Move" else None,
        trajectory_points=traj_pts,
        scene_objects=scene_traces,
        title=f"{scene.name} · 3-DOF Arm",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Status ----
    if st.session_state.status_msg:
        if st.session_state.status_type == "success":
            st.success(st.session_state.status_msg)
        elif st.session_state.status_type == "error":
            st.error(st.session_state.status_msg)
        else:
            st.info(st.session_state.status_msg)

    # ---- Metrics ----
    ee = fk.end_effector_position
    m1, m2, m3 = st.columns(3)
    m1.metric("X (mm)", f"{ee[0]:.1f}")
    m2.metric("Y (mm)", f"{ee[1]:.1f}")
    m3.metric("Z (mm)", f"{ee[2]:.1f}")

    # ---- Joint controls (Manual) ----
    if mode == "Manual Control" and task == "Free Move":
        st.subheader("JOINTS")
        limits_deg = [
            (np.degrees(lo), np.degrees(hi)) for lo, hi in model.joint_limits
        ]
        j1 = st.slider(
            "J1 (base yaw) °",
            float(limits_deg[0][0]),
            float(limits_deg[0][1]),
            float(deg(st.session_state.joint_angles[0])),
            step=1.0,
        )
        j2 = st.slider(
            "J2 (shoulder) °",
            float(limits_deg[1][0]),
            float(limits_deg[1][1]),
            float(deg(st.session_state.joint_angles[1])),
            step=1.0,
        )
        j3 = st.slider(
            "J3 (elbow) °",
            float(limits_deg[2][0]),
            float(limits_deg[2][1]),
            float(deg(st.session_state.joint_angles[2])),
            step=1.0,
        )
        new_angles = np.array([rad(j1), rad(j2), rad(j3)])
        if not np.allclose(new_angles, st.session_state.joint_angles):
            st.session_state.joint_angles = new_angles
            st.session_state.trajectory = None
            st.session_state.traj_history = None
            st.rerun()

        a1, a2, a3 = st.columns(3)
        a1.metric("θ1", f"{j1:.1f}°")
        a2.metric("θ2", f"{j2:.1f}°")
        a3.metric("θ3", f"{j3:.1f}°")

    # ---- Target mode / Reach Target ----
    if mode == "Target Position" or task in ("Reach Target", "Pick & Place"):
        st.subheader("TARGET")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            tx = st.number_input("X (mm)", value=float(st.session_state.target[0]), step=5.0)
        with tcol2:
            ty = st.number_input("Y (mm)", value=float(st.session_state.target[1]), step=5.0)
        with tcol3:
            tz = st.number_input("Z (mm)", value=float(st.session_state.target[2]), step=5.0)
        st.session_state.target = np.array([tx, ty, tz])

        if task == "Pick & Place":
            targets = scene.get_pick_place_targets()
            if st.button("LOAD PICK → PLACE"):
                st.session_state.target = targets["pick"]
                st.session_state.status_msg = "Pick position loaded. Press MOVE TO TARGET, then load place."
                st.session_state.status_type = "info"
                st.rerun()

        if st.button("MOVE TO TARGET", type="primary"):
            _execute_move_to_target(model)

        if st.session_state.last_ik is not None:
            ik = st.session_state.last_ik
            st.write(f"**IK status:** {ik.message}")
            if ik.success:
                st.write(f"Solution (deg): {[round(deg(a), 1) for a in ik.joint_angles]}")
                st.write(f"Position error: {ik.position_error:.3f} mm")

    # ---- Trajectory metrics ----
    if st.session_state.trajectory is not None:
        traj = st.session_state.trajectory
        st.subheader("TRAJECTORY METRICS")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{traj.duration:.2f} s")
        c2.metric("Path length", f"{traj.path_length:.3f} rad")
        c3.metric("Max vel", f"{traj.max_velocity:.3f} rad/s")
        c4.metric("Max acc", f"{traj.max_acceleration:.3f} rad/s²")

        if st.checkbox("Show trajectory graphs", value=False):
            st.plotly_chart(
                create_trajectory_plots(traj),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ---- Workspace ----
    with st.expander("Workspace view"):
        ws = get_workspace_points(model)
        ws_fig = create_workspace_figure(
            model,
            ws,
            current_ee=fk.end_effector_position,
            target=st.session_state.target,
        )
        st.plotly_chart(ws_fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Sensor / PID demo ----
    with st.expander("Sensor & PID foundation"):
        noise = st.checkbox("Sensor noise ON", value=st.session_state.noise_on)
        st.session_state.noise_on = noise
        st.caption(
            "JointPositionSensor adds Gaussian noise when enabled. "
            "PIDController is available for closed-loop experiments."
        )
        if st.button("Demo PID step response"):
            _run_pid_demo(model)

    # ---- Reset ----
    if st.button("RESET"):
        st.session_state.joint_angles = np.zeros(model.joint_count)
        st.session_state.target = np.array([150.0, 0.0, 150.0])
        st.session_state.last_ik = None
        st.session_state.trajectory = None
        st.session_state.traj_history = None
        st.session_state.status_msg = "Reset to home configuration."
        st.session_state.status_type = "info"
        st.rerun()

    st.markdown("---")
    st.caption("Dubotic Lab · Open-source Robotics Simulation · Mathematical core independent of UI")


def _execute_move_to_target(model: RobotModel):
    target = st.session_state.target
    if not is_reachable(model, target):
        st.session_state.status_msg = "Target unreachable (outside geometric workspace)."
        st.session_state.status_type = "error"
        st.session_state.last_ik = None
        return

    ik = inverse_kinematics(model, target)
    st.session_state.last_ik = ik

    if not ik.success:
        st.session_state.status_msg = ik.message
        st.session_state.status_type = "error"
        return

    # Plan trajectory
    planner = TrajectoryPlanner(model, default_duration=2.0)
    traj = planner.plan(st.session_state.joint_angles, ik.joint_angles, n_points=80)
    st.session_state.trajectory = traj

    # Simulate (open-loop ideal)
    sim = Simulation(model=model, state=update_state_from_fk(model, st.session_state.joint_angles))
    history = sim.run_trajectory(traj)
    st.session_state.traj_history = history
    st.session_state.joint_angles = ik.joint_angles.copy()

    st.session_state.status_msg = (
        f"Moved to target. Error = {ik.position_error:.3f} mm · "
        f"Duration = {traj.duration:.2f} s"
    )
    st.session_state.status_type = "success"


def _run_pid_demo(model: RobotModel):
    """Simple closed-loop step to a fixed joint target."""
    start = st.session_state.joint_angles.copy()
    goal = model.clamp_joints(np.array([0.5, 0.3, -0.4]))
    sensor = JointPositionSensor(noise_enabled=st.session_state.noise_on, noise_std=0.02)
    sim = Simulation(
        model=model,
        state=update_state_from_fk(model, start),
        sensor=sensor,
        dt=0.02,
    )
    history = []
    for _ in range(150):
        state = sim.step(goal)
        history.append(state.copy())
        if np.linalg.norm(state.joint_angles - goal) < 0.02:
            break
    st.session_state.traj_history = history
    st.session_state.joint_angles = history[-1].joint_angles
    st.session_state.status_msg = (
        f"PID demo finished in {len(history) * sim.dt:.2f} s "
        f"(noise={'ON' if st.session_state.noise_on else 'OFF'})"
    )
    st.session_state.status_type = "success"


if __name__ == "__main__":
    main()
