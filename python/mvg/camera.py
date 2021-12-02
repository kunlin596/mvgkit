#!/usr/bin/env python3 -B
"""This module includes camera models such as a typical pinhole camera model."""

import numpy as np
import sympy as sp
from enum import IntEnum
from mvg import basic
from collections import namedtuple
from dataclasses import dataclass


# Pinhole camera intrinsics model in OpenCV format
IntrisicsCalibrationData = namedtuple(
    "IntrisicsCalibrationData",
    ["camera_matrix", "dist", "rvecs", "tvecs", "width", "height"],
)


class ProjectionType(IntEnum):
    kPerspective = 0
    kOrthographic = 1


@dataclass
class RadialDistortionModel:
    """Radial distortion model."""

    k1 = 0.0
    k2 = 0.0
    k3 = 0.0

    def distort(self, x, y):
        r2 = x ** 2 + y ** 2
        r4 = r2 ** 2
        r6 = r4 ** 2
        coeff = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        return (x * coeff, y * coeff)


@dataclass
class TangentialDistortionModel:
    """Tangential distortion model."""

    p1 = 0.0
    p2 = 0.0

    def distort(self, x, y):
        r2 = x ** 2 + y ** 2
        xy = x * y
        return (
            x + (2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x ** 2)),
            y + (2.0 * self.p2 * xy + self.p1 * (r2 + 2.0 * y ** 2)),
        )


@dataclass
class CameraMatrix:
    """Pinhole camera matrix, by default it's a canonical camera matrix."""

    fx = 1.0  # Focal length (in pixels) in x direction.
    fy = 1.0  # Focal length (in pixels) in y direction.
    cx = 0.0  # X offset (in pixels) from optical center in image sensor.
    cy = 0.0  # Y offset (in pixels) from optical center in image sensor.
    s = 0.0  # Pixel skew, for rectangular pixel, it should be zero.

    def as_matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, self.s, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def as_symbols(self) -> sp.Matrix:
        return sp.Matrix(
            [
                [sp.Symbol("fx"), sp.Symbol("s"), sp.Symbol("cx")],
                [0.0, sp.Symbol("fy"), sp.Symbol("cy")],
                [0.0, 0.0, 1.0],
            ]
        )

    def from_array(self, array: np.ndarray) -> None:
        assert len(array) == 5
        self.fx = array[0]
        self.fy = array[1]
        self.cx = array[2]
        self.cy = array[3]
        self.s = array[4]

    def from_matrix(self, matrix: np.ndarray) -> None:
        assert matrix.shape == (3, 3)
        self.fx = matrix[0, 0]
        self.fy = matrix[1, 1]
        self.cx = matrix[0, 2]
        self.cy = matrix[1, 2]
        self.s = matrix[0, 1]

    def project(self, points_C: np.ndarray) -> np.ndarray:
        """Project points in camera frame (C) to image plane"""
        image_points = points_C @ self.as_matrix().T
        return (image_points[:, :2] / image_points[:, 2].reshape(-1, 1))[:, :2]

    def unproject(self, image_points: np.ndarray) -> np.ndarray:
        """Unproject points image plane to normalized image plane in 3D."""
        return basic.homogeneous(image_points) @ np.linalg.inv(self.as_matrix()).T
