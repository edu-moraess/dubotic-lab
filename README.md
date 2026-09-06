# Dubotic Lab

**Interactive Robotics Simulation Laboratory**

> Engineering · Robotics · Simulation · Computer Science

Dubotic Lab is an open-source interactive laboratory for studying robotic-arm kinematics, trajectory planning and control.  
It runs entirely in the browser (including Android) via Streamlit and is designed for technical correctness first.

---

## Overview

The first release implements a **3-DOF serial robotic arm** with a clean separation between:

```
ROBOTICS CORE
      ↓
MATHEMATICAL MODEL
      ↓
SIMULATION
      ↓
VISUALIZATION
      ↓
STREAMLIT UI
```

The core library can be imported and used without Streamlit:

```python
from robotics.kinematics import forward_kinematics
from robotics.models import RobotModel

model = RobotModel.default_3dof()
fk = forward_kinematics(model, [0.0, 0.3, -0.4])
print(fk.end_effector_position)
```

---

## Features

- Explicit `RobotModel` and immutable-friendly `RobotState`
- Homogeneous 4×4 transforms with mathematical validation
- Analytic Forward Kinematics
- Analytic Inverse Kinematics + mandatory FK verification
- Joint-space cubic-polynomial trajectory planning
- PID controller with anti-windup and saturation
- Sensor abstraction (joint-position sensor + optional noise)
- Architecture prepared for Kalman filtering
- Three independent scenes: Laboratory, Roblox-style, Tetris-style
- Pick & Place task flow
- Mobile-first Streamlit UI
- MediaPipe Hands com 21 landmarks e overlay visual
- Controle pelo `INDEX_FINGER_TIP` com mapeamento configurável câmera → workspace
- Suavização EMA, gestos MOVE / GRAB / RELEASE / STOP e gates de segurança
- Modo `SIMULATION` explicitamente separado da câmera real
- Full unit-test suite

---

## Architecture

```
dubotic-lab/
├── app.py                  # Streamlit front-end only
├── robotics/               # Pure mathematical / algorithmic core
│   ├── models.py
│   ├── transforms.py
│   ├── kinematics.py
│   ├── inverse_kinematics.py
│   ├── trajectory.py
│   ├── controller.py
│   └── simulation.py
├── visualization/          # Plotly figures (no Streamlit dependency)
├── scenes/                 # Procedural environments
├── vision/                 # MediaPipe, mapping, gestures, EMA and safety
└── tests/                  # pytest suite, including no-camera hand tests
```

---

## Robotics Core

### Robot Model

```
L1 = 150 mm   (vertical column)
L2 = 150 mm   (upper arm)
L3 = 100 mm   (forearm)

J1 ∈ [-180°, +180°]   base yaw   (about Z)
J2 ∈ [ -90°,  +90°]   shoulder   (about Y)
J3 ∈ [-135°, +135°]   elbow      (about Y)
```

### Forward Kinematics

Pipeline:

```
joint angles
     ↓
homogeneous transforms (R_z · Trans_z · R_y · Trans_x · R_y · Trans_x)
     ↓
joint coordinates + end-effector pose
     ↓
visualization
```

### Inverse Kinematics

```
TARGET (x,y,z)
  ↓
geometric reachability check
  ↓
θ1 = atan2(y, x)
  ↓
planar 2-link cosine law → θ2, θ3
  ↓
joint-limit clamp
  ↓
FK verification → position_error
  ↓
success / failure
```

Default verification tolerance: **1 mm**.

### Trajectory Planning

Cubic polynomial per joint with boundary conditions:

```
θ(0)  = θ_start ,  θ̇(0) = 0
θ(T)  = θ_goal  ,  θ̇(T) = 0
```

Produces continuous position, velocity and acceleration profiles.

### PID Control

```
u = Kp·e + Ki·∫e dt + Kd·de/dt
```

with integral anti-windup and output saturation.  
Ready for future motor / actuator models.

### Simulation Loop

```
GROUND TRUTH
     │
     ├──→ SENSOR → MEASUREMENT
     │                 ↓
     │            ESTIMATOR (future)
     │                 ↓
     │         ESTIMATED STATE
     │
     ↓
 CONTROLLER
     ↓
 ACTUATOR (ideal)
     ↓
 ROBOT STATE
```

---

## Scenes

| Scene          | Description                                      |
|----------------|--------------------------------------------------|
| Laboratory     | Technical grid floor, axes, workbench            |
| Roblox-style   | Procedural low-poly blocks (no proprietary assets) |
| Tetris-style   | Procedural tetrominoes for conceptual pick & place |

All scenes consume the same robotics core; only the visual layer changes.

---

## Installation

```bash
git clone https://github.com/<your-user>/dubotic-lab.git
cd dubotic-lab
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running Locally

```bash
streamlit run app.py
```

Open the printed local URL on a desktop or smartphone (same network). Select `Hand Tracking`, allow camera access, frame one hand and capture a frame with `camera_input`. The red marker identifies the index fingertip; the telemetry reports detection, gesture, target XYZ, IK and safety state. `Z` remains the configured fixed working-plane value and is not inferred from an RGB camera.

The gestures are interpreted as follows: one raised index finger is `MOVE`, pinch is `GRAB`, open hand is `RELEASE`, and closed fist is `STOP`. `STOP`, missing hands, invalid landmarks, out-of-workspace targets and IK failures hold the last safe robot position.

For a camera-free demonstration, enable `Demo Mode (synthetic hand)`. It is explicitly labeled `SIMULATION` and does not represent camera input.

---

## Streamlit Cloud

1. Push the repository to GitHub.
2. Connect the repo at https://share.streamlit.io
3. Main file: `app.py`
4. Python version ≥ 3.11

No GPU, no absolute paths, no local-only files.

---

## Testing

```bash
python -m compileall .
PYTHONPATH=. pytest -q
```

The automated suite does not require a real camera; MediaPipe-facing tests use synthetic landmark arrays and blank frames.

---

## Mathematical Model

Homogeneous transform for each joint:

```
T₀₁ = Rot_z(θ₁) · Trans(0, 0, L₁)
T₁₂ = Rot_y(θ₂) · Trans(L₂, 0, 0)
T₂₃ = Rot_y(θ₃) · Trans(L₃, 0, 0)

T₀₃ = T₀₁ · T₁₂ · T₂₃
```

End-effector position = translation part of `T₀₃`.

Rotation matrices are validated:

```
R · Rᵀ ≈ I
det(R) ≈ +1
```

---

## Roadmap

- [x] 3-DOF robotic arm
- [x] Robot state model
- [x] Homogeneous transformations
- [x] Forward kinematics
- [x] Inverse kinematics
- [x] 3D visualization
- [x] Workspace
- [x] Trajectory planning
- [x] Velocity
- [x] Acceleration
- [x] PID foundation
- [x] Laboratory scene
- [x] Roblox-style scene
- [x] Tetris-style scene
- [x] Pick & Place
- [x] Hand tracking control
- [x] Gesture recognition (MOVE/GRAB/RELEASE/STOP)
- [x] Coordinate mapping (pixel → workspace)
- [x] EMA smoothing
- [ ] 4-DOF
- [ ] 6-DOF
- [ ] DH parameter editor
- [ ] Jacobian
- [ ] Singularity detection
- [ ] Dynamic model
- [ ] Motor model
- [ ] Encoder simulation
- [ ] IMU simulation
- [ ] Kalman Filter
- [ ] Sensor fusion
- [ ] Computer Vision
- [ ] Object detection
- [ ] Path planning
- [ ] SLAM
- [ ] ROS integration
- [ ] Digital Twin

---

## License

MIT License – see [LICENSE](LICENSE).

---

**Dubotic Lab** – built for engineers who want real mathematics inside an interactive simulation.

---

## Hand Tracking (v0.2)

Control the robotic arm with your hand via the device camera.

```
CAMERA
  ↓
HAND TRACKING (MediaPipe Hands — 21 landmarks)
  ↓
INDEX FINGER TIP
  ↓
COORDINATE MAPPING  (pixel → workspace plane)
  ↓
SMOOTHING (EMA)
  ↓
TARGET XYZ
  ↓
EXISTING IK → TRAJECTORY → PID → ROBOT
```

### How it works

1. Enable **CONTROL MODE → Hand Tracking**.
2. Allow camera access (or use **Demo Mode** with synthetic landmarks).
3. Point the index finger; its tip becomes the spatial target.
4. The existing Inverse Kinematics solves for joint angles.
5. The 3-D arm follows the target.

### Gestures (geometry-based)

| Gesture  | Meaning                         |
|----------|---------------------------------|
| ☝️ MOVE  | Index extended → follow finger  |
| 🤏 GRAB  | Pinch → prepare grasp           |
| 🖐️ RELEASE | Open hand → release           |
| ✊ STOP  | Fist → hold / interrupt motion  |

### Coordinate mapping

A single RGB camera does **not** provide absolute metric depth.
v0.2 maps image coordinates `(u, v)` onto a configurable **working plane**
with fixed `Z = workspace_height` via an affine transform defined by
`CameraBounds` ↔ `WorkspaceBounds`.

This is explicitly a 2.5-D solution, not true stereo / depth reconstruction.

### Smoothing

Exponential Moving Average (EMA) reduces landmark jitter before IK.

### Demo / Simulation mode

When no camera is available (CI, Streamlit Cloud without permission, etc.):

- Enable **Demo Mode (synthetic hand)**
- Landmarks are generated procedurally and clearly labelled **SIMULATION**

### Architecture

```
vision/
├── coordinate_mapping.py   # pixel ↔ workspace
├── hand_tracking.py        # MediaPipe adapter
├── gestures.py             # rule-based classifier
├── smoothing.py            # EMA filter
└── demo.py                 # synthetic landmarks
```

### Limitations (documented)

- Z is fixed to the working plane (relative MediaPipe depth is not treated as metric).
- Continuous video streaming requires browser camera permission; `st.camera_input` captures frames on demand (Cloud-compatible).
- MediaPipe must be installed; if unavailable the tracker backend reports `none` and Demo Mode remains usable.

### Requirements added in v0.2

```
opencv-python-headless
mediapipe
Pillow
```

---

## Architecture (full pipeline)

```
Camera / Demo
     ↓
Hand Tracking (21 landmarks)
     ↓
Gesture + Index Tip
     ↓
Coordinate Mapping
     ↓
EMA Smoothing
     ↓
Target XYZ
     ↓
Inverse Kinematics
     ↓
Trajectory Planner
     ↓
PID / Simulation
     ↓
3-D Robot Visualisation
```

