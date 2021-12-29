#!/usr/bin/env python3 -B
"""This module includes camera models such as a typical pinhole camera model."""

from dataclasses import dataclass
from typing import Optional
from enum import IntEnum

import cv2
import numpy as np
import sympy as sym

from mvg.basic import SE3, homogeneous


class ProjectionType(IntEnum):
    kPerspective = 0
    kOrthographic = 1


@dataclass
class RadialDistortionModel:
    """Radial distortion model."""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    def distort(self, *, normalized_image_points: np.ndarray) -> np.ndarray:
        x = normalized_image_points[:, 0]
        y = normalized_image_points[:, 1]
        r2 = np.linalg.norm(normalized_image_points, axis=1) ** 2
        r4 = r2 ** 2
        r6 = r4 ** 2
        coeff = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        return np.vstack([x * coeff, y * coeff]).T

    def as_array(self):
        return np.asarray([self.k1, self.k2, self.k3])


@dataclass
class TangentialDistortionModel:
    """Tangential distortion model."""

    p1: float = 0.0
    p2: float = 0.0

    def distort(self, normalized_image_points: np.ndarray) -> np.ndarray:
        r2 = np.linalg.norm(normalized_image_points, axis=1) ** 2
        x = normalized_image_points[:, 0]
        y = normalized_image_points[:, 1]
        xy = x * y
        return np.vstack(
            [
                x + (2.0 * x * xy + y * (r2 + 2.0 * x ** 2)),
                y + (2.0 * y * xy + x * (r2 + 2.0 * y ** 2)),
            ]
        ).T


def distort(*, image_points: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
    assert (
        len(distortion_coeffs) == 5
    ), "Only support distortion coefficients in the format of [k1, k2, k3, p1, p2]!"
    distorted = RadialDistortionModel(distortion_coeffs[:3]).distort(image_points)
    distorted = TangentialDistortionModel(distortion_coeffs[3:]).distort(distorted)
    return distorted


@dataclass
class CameraMatrix:
    """Pinhole camera matrix, by default it's a canonical camera matrix."""

    fx: float = 1.0  # Focal length (in pixels) in x direction.
    fy: float = 1.0  # Focal length (in pixels) in y direction.
    cx: float = 0.0  # X offset (in pixels) from optical center in image sensor.
    cy: float = 0.0  # Y offset (in pixels) from optical center in image sensor.
    s: float = 0.0  # Pixel skew, for rectangular pixel, it should be zero.

    def as_array(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.cx, self.cy, self.s])

    def as_matrix(self) -> np.ndarray:
        return np.asarray([[self.fx, self.s, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def as_symbols(self) -> sym.Matrix:
        return sym.Matrix(
            [
                [sym.Symbol("fx"), sym.Symbol("s"), sym.Symbol("cx")],
                [0.0, sym.Symbol("fy"), sym.Symbol("cy")],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def from_array(array: np.ndarray):
        assert len(array) == 5
        return CameraMatrix(array[0], array[1], array[2], array[3], array[4])

    @staticmethod
    def from_matrix(matrix: np.ndarray):
        assert matrix.shape == (3, 3)
        return CameraMatrix(matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2], matrix[0, 1])

    def project(self, points_C: np.ndarray) -> np.ndarray:
        """Project points in camera frame (C) to image plane"""
        image_points = points_C @ self.as_matrix().T
        return (image_points[:, :2] / image_points[:, 2].reshape(-1, 1))[:, :2]

    def unproject(self, image_points: np.ndarray) -> np.ndarray:
        """Unproject points image plane to normalized image plane in 3D."""
        return homogeneous(image_points) @ np.linalg.inv(self.as_matrix()).T

    @staticmethod
    def project_to_normalized_image_plane(*, points_C):
        points_C = points_C.copy()
        points_C[:, :2] /= points_C[:, -1].reshape(-1, 1)
        return points_C[:, :2]

    def project_to_sensor_image_plane(self, *, normalized_image_points):
        K = self.as_matrix()
        return normalized_image_points @ K[:2, :2].T + K[:2, -1]


def project_points(
    *,
    object_points_W: np.ndarray,
    camera_matrix: CameraMatrix,
    T_CW: Optional[SE3] = None,
    radial_distortion_model: Optional[RadialDistortionModel] = None,
) -> np.ndarray:
    if T_CW is None:
        T_CW = SE3()

    assert object_points_W is not None

    object_points_C = object_points_W @ T_CW.R.as_matrix().T + T_CW.t

    if radial_distortion_model is not None:
        normalized_image_points = camera_matrix.project_to_normalized_image_plane(
            points_C=object_points_C
        )
        distorted_normalized_image_points = radial_distortion_model.distort(
            normalized_image_points=normalized_image_points
        )
        projected = camera_matrix.project_to_sensor_image_plane(
            normalized_image_points=distorted_normalized_image_points
        )
    else:
        projected = camera_matrix.project(object_points_C)
    return projected


def undistort(
    *,
    image: np.ndarray,
    camera_matrix: CameraMatrix,
    radial_distortion_model: Optional[RadialDistortionModel] = None,
):
    shape = image.shape
    undistorted_image_points = np.mgrid[: shape[1], : shape[0]].T
    undistorted_normalized_points = camera_matrix.unproject(undistorted_image_points.reshape(-1, 2))

    distorted_normalized_points = undistorted_normalized_points
    if radial_distortion_model is not None:
        distorted_normalized_points = radial_distortion_model.distort(
            normalized_image_points=undistorted_normalized_points
        )

    distorted_image_points = camera_matrix.project_to_sensor_image_plane(
        normalized_image_points=distorted_normalized_points
    )

    distorted_image_points_map = distorted_image_points.reshape(shape[0], shape[1], -1)
    mapx = distorted_image_points_map[:, :, 0].astype(np.float32, copy=False)
    mapy = distorted_image_points_map[:, :, 1].astype(np.float32, copy=False)
    undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return undistorted_image


@dataclass
class PinholeCamera:
    """
    This class describes a pinhole camera model.
    """

    K: Optional[CameraMatrix] = None
    k: Optional[RadialDistortionModel] = None
    p: Optional[TangentialDistortionModel] = None
    T: Optional[SE3] = None  # Extrinsics, transforms the points in reference frame to camera frame.

    def __post_init__(self):
        if self.K is None:
            self.K = CameraMatrix()
        if self.T is None:
            self.T = SE3()

    @property
    def P(self):
        """Camera projection matrix."""
        return self.K.as_matrix() @ self.T.as_augmented_matrix()
