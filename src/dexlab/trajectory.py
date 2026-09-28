"""Simple Cartesian trajectory generation and inverse-kinematics tracking."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .planar_arm import PlanarArm


@dataclass(frozen=True)
class TrackingResult:
    desired_positions: NDArray[np.float64]
    actual_positions: NDArray[np.float64]
    joint_positions: NDArray[np.float64]
    all_converged: bool

    @property
    def errors(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.desired_positions - self.actual_positions, axis=1)

    @property
    def maximum_error(self) -> float:
        return float(np.max(self.errors))


def linear_trajectory(start: ArrayLike, goal: ArrayLike, steps: int) -> NDArray[np.float64]:
    start_position = np.asarray(start, dtype=float)
    goal_position = np.asarray(goal, dtype=float)
    if start_position.shape != (2,) or goal_position.shape != (2,):
        raise ValueError("start and goal must be two-dimensional positions")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    return np.linspace(start_position, goal_position, steps)


def track_cartesian_trajectory(
    arm: PlanarArm,
    desired_positions: ArrayLike,
    initial_joint_positions: ArrayLike,
) -> TrackingResult:
    targets = np.asarray(desired_positions, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("desired_positions must have shape (steps, 2)")

    current = np.asarray(initial_joint_positions, dtype=float).copy()
    joint_history = []
    actual_history = []
    all_converged = True

    for target in targets:
        result = arm.inverse_kinematics(target, current)
        current = result.joint_positions
        joint_history.append(current.copy())
        actual_history.append(arm.forward_kinematics(current))
        all_converged = all_converged and result.converged

    return TrackingResult(
        targets,
        np.asarray(actual_history),
        np.asarray(joint_history),
        all_converged,
    )


