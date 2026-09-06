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
└── tests/                  # pytest suite
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

Open the printed local URL on a desktop or smartphone (same network).

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
pytest -q
python -m compileall .
```

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
