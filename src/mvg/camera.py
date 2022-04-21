"""This module includes camera models such as a typical pinhole camera model.

TODO: Check for tilt distortion.
TODO: Implement orthographic projection
TODO: Implement image (un)-projection
"""

from dataclasses import dataclass
from enum import IntEnum
from mvg import image_processing
from typing import Optional

import cv2
import numpy as np

from mvg.basic import SE3, homogenize


class ProjectionType(IntEnum):
    kPerspective = 0
    kOrthographic = 1


@dataclass
class RadialDistortionModel:
    """Radial distortion model."""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    # TODO: k4, k5, k6 are not used yet.
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0

    def get_coord_coeffs(self, image_points):
        r2 = np.linalg.norm(image_points, axis=1) ** 2
        r4 = r2**2
        r6 = r2 * r4
        coeffs = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        return coeffs.reshape(len(image_points), 1)

    def as_array(self):
        return np.asarray([self.k1, self.k2, self.k3, self.k3, self.k5, self.k6])


@dataclass
class TangentialDistortionModel:
    """Tangential distortion model."""

    p1: float = 0.0
    p2: float = 0.0

    def get_coord_coeffs(self, image_points):
        r2 = np.linalg.norm(image_points, axis=1) ** 2
        x = image_points[:, 0]
        y = image_points[:, 1]
        xy = x * y
        delta_x = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x**2)
        delta_y = 2.0 * self.p2 * xy + self.p1 * (r2 + 2.0 * y**2)
        return np.vstack([delta_x, delta_y]).T

    def as_array(self):
        return np.asarray([self.p1, self.p2])


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

    @staticmethod
    def from_array(array: np.ndarray):
        assert len(array) == 5
        return CameraMatrix(array[0], array[1], array[2], array[3], array[4])

    @staticmethod
    def from_matrix(matrix: np.ndarray):
        assert matrix.shape == (3, 3)
        return CameraMatrix(matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2], matrix[0, 1])

    def project(self, points_C: np.ndarray) -> np.ndarray:
        """Project points in camera frame (C) to image plane."""
        image_points = points_C @ self.as_matrix().T
        return (image_points[:, :2] / image_points[:, 2].reshape(-1, 1))[:, :2]

    def unproject(self, image_points: np.ndarray) -> np.ndarray:
        """Unproject points image plane to normalized image plane in 3D."""
        return homogenize(image_points) @ np.linalg.inv(self.as_matrix()).T

    @staticmethod
    def project_to_normalized_image_plane(*, points_C):
        points_C = points_C.copy()
        points_C[:, :2] /= points_C[:, -1].reshape(-1, 1)
        return points_C[:, :2]

    def project_to_sensor_image_plane(self, *, normalized_image_points):
        K = self.as_matrix()
        return normalized_image_points @ K[:2, :2].T + K[:2, -1]

    def unproject_to_normalized_image_plane(self, sensor_image_points):
        K = self.as_matrix()
        return (sensor_image_points - K[:2, -1]) @ np.linalg.inv(K[:2, :2]).T


@dataclass
class Camera:
    """
    This class describes a pinhole camera model.
    """

    K: Optional[CameraMatrix] = CameraMatrix()
    k: Optional[RadialDistortionModel] = RadialDistortionModel()
    p: Optional[TangentialDistortionModel] = TangentialDistortionModel()

    # Extrinsics, transforms the points in reference frame to camera frame.
    T: Optional[SE3] = SE3()

    def __post_init__(self):
        self._P = self.K.as_matrix() @ self.T.as_augmented_matrix()

    @property
    def P(self):
        """Camera projection matrix."""
        return self._P

    @staticmethod
    def from_projection_matrix(self):
        pass

    def project_points(self, points_W, distort=False):
        """Project points in world frame to image points."""
        points_C = (self.T @ homogenize(points_W))[:, :3]
        image_points = self.K.project(points_C=points_C)
        if distort:
            return self.distort_points(image_points)
        return image_points

    def distort_points(self, undistorted_image_points: np.ndarray):
        """Distort the undistorted image points."""
        normalized_image_points = self.K.unproject_to_normalized_image_plane(
            undistorted_image_points
        )
        k_coeffs = self.k.get_coord_coeffs(normalized_image_points)
        p_coeffs = self.p.get_coord_coeffs(normalized_image_points)
        distorted = normalized_image_points * k_coeffs + p_coeffs
        image_points = self.K.project_to_sensor_image_plane(normalized_image_points=distorted)
        return image_points

    def undistort_points(self, distorted_image_points: np.ndarray):
        """Undistort the distorted image points."""
        normalized_image_points = self.K.unproject_to_normalized_image_plane(distorted_image_points)
        k_coeffs = self.k.get_coord_coeffs(normalized_image_points)
        p_coeffs = self.p.get_coord_coeffs(normalized_image_points)
        undistorted = (normalized_image_points - p_coeffs) / k_coeffs
        image_points = self.K.project_to_sensor_image_plane(normalized_image_points=undistorted)
        return image_points

    def get_distortion_coefficients(self):
        k_coeffs = self.k.as_array()
        p_coeffs = self.p.as_array()
        return np.asarray([k_coeffs[0], k_coeffs[1], p_coeffs[0], p_coeffs[1], k_coeffs[2]])

    def get_optimal_camera_matrix(self, src_image_size, dest_image_size, alpha=1.0):
        """Get the optimal matrix that can remove the unwanted black pixels after applying undistortion."""
        raw_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.K.as_matrix(),
            distCoeffs=self.get_distortion_coefficients(),
            imageSize=src_image_size,
            alpha=alpha,
            newImgSize=dest_image_size,
        )

        return CameraMatrix.from_matrix(raw_camera_matrix), roi

    def undistort_image(self, image: image_processing.Image, alpha: float = 1.0, crop=False):
        new_K, roi = self.get_optimal_camera_matrix(image.size, image.size, alpha)
        undistorted = cv2.undistort(
            image.data,
            self.K.as_matrix(),
            self.get_distortion_coefficients(),
            None,
            new_K.as_matrix(),
        )
        if crop:
            x, y, w, h = roi
            undistorted = undistorted[y : y + h, x : x + w]
        return image_processing.Image(undistorted, image.timestamp)
