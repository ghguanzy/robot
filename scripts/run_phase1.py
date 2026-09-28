"""Run the phase-one demonstration and print measurable acceptance results."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexlab import PlanarArm, linear_trajectory, track_cartesian_trajectory  # noqa: E402


def main() -> int:
    arm = PlanarArm([0.45, 0.35, 0.20])
    initial = np.deg2rad([35.0, -55.0, 30.0])
    start = arm.forward_kinematics(initial)
    goal = np.array([0.55, 0.35])

    direct = arm.inverse_kinematics(goal, initial)
    path = linear_trajectory(start, goal, 40)
    tracking = track_cartesian_trajectory(arm, path, initial)
    unreachable = arm.inverse_kinematics([1.25, 0.0], initial, max_iterations=200)

    print("Phase 1: kinematics and numerical IK")
    print(f"DOF: {arm.dof}; maximum reach: {arm.maximum_reach:.3f} m")
    print(
        "Reachable target: "
        f"converged={direct.converged}, iterations={direct.iterations}, "
        f"error={direct.position_error:.8f} m"
    )
    print(
        "Path tracking: "
        f"all_converged={tracking.all_converged}, "
        f"maximum_error={tracking.maximum_error:.8f} m"
    )
    print(
        "Unreachable target: "
        f"correctly_rejected={not unreachable.converged}, "
        f"residual_error={unreachable.position_error:.4f} m"
    )

    passed = (
        direct.converged
        and direct.position_error < 1e-4
        and tracking.all_converged
        and tracking.maximum_error < 1e-4
        and not unreachable.converged
    )
    print("Acceptance:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


