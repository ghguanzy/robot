"""Small, dependency-light exercises for dexterous manipulation study."""

from .planar_arm import IKResult, PlanarArm
from .trajectory import linear_trajectory, track_cartesian_trajectory

__all__ = ["IKResult", "PlanarArm", "linear_trajectory", "track_cartesian_trajectory"]


