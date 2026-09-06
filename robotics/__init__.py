"""
Dubotic Lab - Robotics Core

Mathematical and algorithmic foundation for robotic arm simulation.
Independent of UI frameworks.
"""

from .models import RobotModel, RobotState
from .transforms import (
    rotation_x,
    rotation_y,
    rotation_z,
    translation,
    homogeneous_transform,
    validate_rotation,
)
from .kinematics import forward_kinematics, compute_joint_positions
from .inverse_kinematics import inverse_kinematics, is_reachable
from .trajectory import TrajectoryPlanner, Trajectory
from .controller import PIDController
from .simulation import Simulation, Sensor

__all__ = [
    "RobotModel",
    "RobotState",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "translation",
    "homogeneous_transform",
    "validate_rotation",
    "forward_kinematics",
    "compute_joint_positions",
    "inverse_kinematics",
    "is_reachable",
    "TrajectoryPlanner",
    "Trajectory",
    "PIDController",
    "Simulation",
    "Sensor",
]

__version__ = "0.1.0"
