from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexlab import PlanarArm, linear_trajectory, track_cartesian_trajectory  # noqa: E402


class PlanarArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = PlanarArm([0.45, 0.35, 0.20])

    def test_forward_kinematics_at_zero(self) -> None:
        actual = self.arm.forward_kinematics(np.zeros(3))
        np.testing.assert_allclose(actual, [1.0, 0.0], atol=1e-12)

    def test_analytic_jacobian_matches_finite_difference(self) -> None:
        q = np.array([0.4, -0.7, 0.3])
        epsilon = 1e-7
        numerical = np.zeros((2, self.arm.dof))
        for index in range(self.arm.dof):
            offset = np.zeros(self.arm.dof)
            offset[index] = epsilon
            numerical[:, index] = (
                self.arm.forward_kinematics(q + offset)
                - self.arm.forward_kinematics(q - offset)
            ) / (2 * epsilon)
        np.testing.assert_allclose(self.arm.jacobian(q), numerical, atol=1e-7)

    def test_inverse_kinematics_reaches_target(self) -> None:
        result = self.arm.inverse_kinematics(
            [0.55, 0.35], np.deg2rad([35.0, -55.0, 30.0])
        )
        self.assertTrue(result.converged)
        self.assertLess(result.position_error, 1e-4)

    def test_unreachable_target_does_not_report_success(self) -> None:
        result = self.arm.inverse_kinematics([1.25, 0.0], [0.2, -0.1, 0.1])
        self.assertFalse(result.converged)
        self.assertGreater(result.position_error, 0.2)

    def test_cartesian_path_tracking(self) -> None:
        initial = np.deg2rad([35.0, -55.0, 30.0])
        start = self.arm.forward_kinematics(initial)
        path = linear_trajectory(start, [0.55, 0.35], 40)
        result = track_cartesian_trajectory(self.arm, path, initial)
        self.assertTrue(result.all_converged)
        self.assertLess(result.maximum_error, 1e-4)


if __name__ == "__main__":
    unittest.main()


