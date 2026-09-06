"""
Simulation engine separating ground-truth, sensors and control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .models import RobotModel, RobotState
from .kinematics import forward_kinematics, update_state_from_fk
from .controller import PIDController
from .trajectory import Trajectory


class Sensor(ABC):
    """Abstract sensor interface."""

    @abstractmethod
    def measure(self, true_state: RobotState) -> np.ndarray:
        """Return a measurement derived from ground-truth state."""
        ...


@dataclass
class JointPositionSensor(Sensor):
    """
    Simulated joint-position sensor (encoder-like).

    When noise_enabled=True adds zero-mean Gaussian noise.
    """

    noise_std: float = 0.01  # rad
    noise_enabled: bool = False
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )

    def measure(self, true_state: RobotState) -> np.ndarray:
        angles = true_state.joint_angles.copy()
        if self.noise_enabled and self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, size=angles.shape)
            angles = angles + noise
        return angles


@dataclass
class Simulation:
    """
    Lightweight discrete-time simulation loop.

    Architecture (current + future-ready):

        GROUND TRUTH
             │
             ├──→ SENSOR → MEASUREMENT
             │                 ↓
             │            ESTIMATOR (future Kalman)
             │                 ↓
             │         ESTIMATED STATE
             │
             ↓
         CONTROLLER
             ↓
         ACTUATOR (ideal for now)
             ↓
         ROBOT STATE
    """

    model: RobotModel
    state: RobotState
    controller: Optional[PIDController] = None
    sensor: Optional[Sensor] = None
    dt: float = 0.01
    time: float = 0.0

    def __post_init__(self):
        if self.sensor is None:
            self.sensor = JointPositionSensor(noise_enabled=False)
        if self.controller is None:
            n = self.model.joint_count
            self.controller = PIDController(
                Kp=np.full(n, 8.0),
                Ki=np.full(n, 0.5),
                Kd=np.full(n, 0.8),
                output_limits=(-2.0, 2.0),  # rad/s command limit
            )

    def reset(self, joint_angles: Optional[np.ndarray] = None) -> None:
        if joint_angles is None:
            joint_angles = np.zeros(self.model.joint_count)
        self.state = update_state_from_fk(self.model, joint_angles)
        self.time = 0.0
        if self.controller:
            self.controller.reset()

    def step(self, target_angles: np.ndarray) -> RobotState:
        """
        Advance one simulation step toward target joint angles.
        Ideal kinematics + simple velocity integration.
        """
        measurement = self.sensor.measure(self.state)
        control = self.controller.update(target_angles, measurement, self.dt)

        # Simple integrator: treat control as velocity command
        new_vel = control
        new_angles = self.state.joint_angles + new_vel * self.dt
        new_angles = self.model.clamp_joints(new_angles)

        self.state = update_state_from_fk(
            self.model,
            new_angles,
            velocities=new_vel,
            timestamp=self.time + self.dt,
        )
        self.time += self.dt
        return self.state

    def run_trajectory(
        self, trajectory: Trajectory, record: bool = True
    ) -> list[RobotState]:
        """Execute a pre-planned trajectory open-loop (joint positions)."""
        history = []
        for i, t in enumerate(trajectory.time):
            q = trajectory.position[i]
            # Open-loop: directly set joints (ideal)
            self.state = update_state_from_fk(
                self.model,
                q,
                velocities=trajectory.velocity[i],
                accelerations=trajectory.acceleration[i],
                timestamp=t,
            )
            self.time = t
            if record:
                history.append(self.state.copy())
        return history
