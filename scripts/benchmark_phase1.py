"""Deterministic phase-one benchmark for the roadmap promotion criteria."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexlab import PlanarArm  # noqa: E402


def finite_difference_jacobian(
    arm: PlanarArm, joint_positions: np.ndarray, epsilon: float = 1e-7
) -> np.ndarray:
    numerical = np.zeros((2, arm.dof))
    for index in range(arm.dof):
        offset = np.zeros(arm.dof)
        offset[index] = epsilon
        numerical[:, index] = (
            arm.forward_kinematics(joint_positions + offset)
            - arm.forward_kinematics(joint_positions - offset)
        ) / (2 * epsilon)
    return numerical


def main() -> int:
    arm = PlanarArm([0.45, 0.35, 0.20])
    rng = np.random.default_rng(42)

    maximum_jacobian_error = 0.0
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, arm.dof)
        error = np.max(np.abs(arm.jacobian(q) - finite_difference_jacobian(arm, q)))
        maximum_jacobian_error = max(maximum_jacobian_error, float(error))

    initial_states = [
        np.array([0.4, -0.6, 0.2]),
        np.array([-1.0, 0.8, 0.5]),
        np.array([1.2, -1.1, 0.7]),
    ]
    successes = 0
    attempts = 0
    maximum_ik_error = 0.0
    for _ in range(20):
        target_q = rng.uniform(-np.pi, np.pi, arm.dof)
        target = arm.forward_kinematics(target_q)
        for initial in initial_states:
            result = arm.inverse_kinematics(target, initial, max_iterations=1000)
            attempts += 1
            successes += int(result.converged)
            maximum_ik_error = max(maximum_ik_error, result.position_error)

    success_rate = successes / attempts
    passed = maximum_jacobian_error < 1e-5 and success_rate >= 0.95

    print("Phase 1 promotion benchmark")
    print(f"Numerical Jacobian max error: {maximum_jacobian_error:.3e}")
    print(f"Random-target IK: {successes}/{attempts} ({success_rate:.1%})")
    print(f"Maximum final IK error: {maximum_ik_error:.3e} m")
    print("Promotion gate:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


