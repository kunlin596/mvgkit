#!/usr/bin/env python3
"""
FIXME: The tests below are incomplete.
"""

import unittest
import numpy as np
from mvg.camera import CameraMatrix, RadialDistortionModel, TangentialDistortionModel


class CameraTest(unittest.TestCase):
    @property
    def camera_matrix(self):
        return CameraMatrix(
            fx=535.35887264164,
            fy=535.6467965086524,
            cx=342.63131029730295,
            cy=233.77551538388735,
            s=0.5369513302298261,
        )

    @property
    def radial_distortion(self):
        return RadialDistortionModel(-0.4, 0.5, -0.57)

    @property
    def tangential_distortion(self):
        return TangentialDistortionModel(0.01, 0.05)

    @property
    def rng(self):
        return np.random.default_rng(42)

    @property
    def num_points(self):
        return 10

    @property
    def points_C(self):
        points_C = np.dstack(
            np.meshgrid(np.linspace(-1, 1, self.num_points), np.linspace(-1, 1, self.num_points))
        )
        points_C = points_C.reshape(-1, 2)
        points_C = np.hstack([points_C, np.ones((len(points_C), 1)) * 2.0])
        return points_C


if __name__ == "__main__":
    unittest.main()
