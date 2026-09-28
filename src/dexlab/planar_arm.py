"""Planar serial-arm kinematics used by the first learning phase.

The model is deliberately small enough that every formula can be derived by
hand, while retaining the numerical issues found in larger robot arms.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class IKResult:
    joint_positions: Vector
    converged: bool
    iterations: int
    position_error: float


class PlanarArm:
    """An n-link planar revolute arm."""

    def __init__(self, link_lengths: ArrayLike):
        lengths = np.asarray(link_lengths, dtype=float)
        if lengths.ndim != 1 or lengths.size == 0:
            raise ValueError("link_lengths must be a non-empty one-dimensional array")
        if np.any(lengths <= 0):
            raise ValueError("all link lengths must be positive")
        self.link_lengths = lengths

    @property
    def dof(self) -> int:
        return int(self.link_lengths.size)

    @property
    def maximum_reach(self) -> float:
        return float(np.sum(self.link_lengths))

    def _validate_joints(self, joint_positions: ArrayLike) -> Vector:
        q = np.asarray(joint_positions, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"expected {self.dof} joint positions, got shape {q.shape}")
        return q

    def joint_points(self, joint_positions: ArrayLike) -> NDArray[np.float64]:
        """Return base, joint and end-effector coordinates with shape (n+1, 2)."""
        q = self._validate_joints(joint_positions)
        cumulative_angles = np.cumsum(q)
        segments = np.column_stack(
            (
                self.link_lengths * np.cos(cumulative_angles),
                self.link_lengths * np.sin(cumulative_angles),
            )
        )
        return np.vstack((np.zeros(2), np.cumsum(segments, axis=0)))

    def forward_kinematics(self, joint_positions: ArrayLike) -> Vector:
        """Return the end-effector x/y position."""
        return self.joint_points(joint_positions)[-1]

    def jacobian(self, joint_positions: ArrayLike) -> NDArray[np.float64]:
        """Return the analytic 2-by-n translational Jacobian."""
        q = self._validate_joints(joint_positions)
        cumulative_angles = np.cumsum(q)
        jacobian = np.zeros((2, self.dof), dtype=float)
        for joint_index in range(self.dof):
            affected_lengths = self.link_lengths[joint_index:]
            affected_angles = cumulative_angles[joint_index:]
            jacobian[0, joint_index] = -np.sum(
                affected_lengths * np.sin(affected_angles)
            )
            jacobian[1, joint_index] = np.sum(
                affected_lengths * np.cos(affected_angles)
            )
        return jacobian

    def inverse_kinematics(
        self,
        target: ArrayLike,
        initial_joint_positions: ArrayLike | None = None,
        *,
        damping: float = 0.05,
        step_size: float = 0.7,
        tolerance: float = 1e-5,
        max_iterations: int = 500,
    ) -> IKResult:
        """Solve position IK using the damped-least-squares Jacobian inverse.

        The update is dq = J^T (J J^T + lambda^2 I)^-1 error.  Damping keeps
        the update bounded near singular configurations.
        """
        desired = np.asarray(target, dtype=float)
        if desired.shape != (2,):
            raise ValueError("target must contain x and y")
        if damping <= 0 or not 0 < step_size <= 1 or tolerance <= 0:
            raise ValueError("damping/tolerance must be positive and step_size in (0, 1]")

        q = (
            np.zeros(self.dof, dtype=float)
            if initial_joint_positions is None
            else self._validate_joints(initial_joint_positions).copy()
        )
        identity = np.eye(2)

        for iteration in range(max_iterations + 1):
            error = desired - self.forward_kinematics(q)
            error_norm = float(np.linalg.norm(error))
            if error_norm <= tolerance:
                return IKResult(q, True, iteration, error_norm)
            if iteration == max_iterations:
                break
            jacobian = self.jacobian(q)
            regularized = jacobian @ jacobian.T + damping**2 * identity
            delta = jacobian.T @ np.linalg.solve(regularized, error)
            q += step_size * delta
            q = (q + np.pi) % (2 * np.pi) - np.pi

        final_error = float(np.linalg.norm(desired - self.forward_kinematics(q)))
        return IKResult(q, False, max_iterations, final_error)


