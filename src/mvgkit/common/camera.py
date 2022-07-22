"""This module includes camera models such as a typical pinhole camera model.

TODO: Check for tilt distortion.
TODO: Implement orthographic projection
TODO: Implement image (un)-projection
"""

from enum import IntEnum


class ProjectionType(IntEnum):
    kPerspective = 0
    kOrthographic = 1


from pymvgkit_common import (  # noqa
    Camera,
    CameraMatrix,
    RadialDistortionModel,
    TangentialDistortionModel,
)

# @dataclass
# class PyCamera:
#     """
#     This class describes a pinhole camera model.
#     """

#     K: CameraMatrix
#     k: Optional[RadialDistortionModel] = RadialDistortionModel()
#     p: Optional[TangentialDistortionModel] = TangentialDistortionModel()
#     image_size: Optional[Tuple[int]] = (0, 0)

#     # Extrinsics, transforms the points in reference frame to camera frame.
#     T: Optional[SE3] = SE3()

#     def __post_init__(self):
#         self._P = self.K.as_matrix() @ self.T.as_augmented_matrix()

#     @property
#     def P(self):
#         """Camera projection matrix."""
#         return self._P

#     @staticmethod
#     def from_projection_matrix(self):
#         pass

#     def project_points(self, points_W, distort=False):
#         """Project points in world frame to image points."""
#         points_C = (self.T @ homogenize(points_W))[:, :3]
#         image_points = self.K.project(points_C)
#         if distort:
#             return self.distort_points(image_points)
#         return image_points

#     def distort_points(self, undistorted_image_points: np.ndarray):
#         """Distort the undistorted image points."""
#         normalized_image_points = self.K.unproject_to_normalized_image_plane(
#             undistorted_image_points
#         )
#         k_coeffs = self.k.get_coord_coeffs(normalized_image_points)
#         p_coeffs = self.p.get_coord_coeffs(normalized_image_points)
#         distorted = normalized_image_points * k_coeffs + p_coeffs
#         image_points = self.K.project_to_sensor_image_plane(normalized_image_points=distorted)
#         return image_points

#     def undistort_points(self, distorted_image_points: np.ndarray):
#         """Undistort the distorted image points."""
#         normalized_image_points = self.K.unproject_to_normalized_image_plane(distorted_image_points)
#         k_coeffs = self.k.get_coord_coeffs(normalized_image_points)
#         p_coeffs = self.p.get_coord_coeffs(normalized_image_points)
#         undistorted = (normalized_image_points - p_coeffs) / k_coeffs
#         image_points = self.K.project_to_sensor_image_plane(normalized_image_points=undistorted)
#         return image_points

#     def get_distortion_coefficients(self):
#         k_coeffs = self.k.as_array()
#         p_coeffs = self.p.as_array()
#         return np.asarray([k_coeffs[0], k_coeffs[1], p_coeffs[0], p_coeffs[1], k_coeffs[2]])

#     def get_optimal_camera_matrix(self, src_image_size, dest_image_size, alpha=1.0):
#         """Get the optimal matrix that can remove the unwanted black pixels after applying undistortion."""
#         raw_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
#             cameraMatrix=self.K.as_matrix(),
#             distCoeffs=self.get_distortion_coefficients(),
#             imageSize=src_image_size,
#             alpha=alpha,
#             newImgSize=dest_image_size,
#         )

#         return CameraMatrix.from_matrix(raw_camera_matrix), roi

#     def undistort_image(self, image: image_processing.Image, alpha: float = 1.0, crop=False):
#         new_K, roi = self.get_optimal_camera_matrix(image.size, image.size, alpha)
#         undistorted = cv2.undistort(
#             image.data,
#             self.K.as_matrix(),
#             self.get_distortion_coefficients(),
#             None,
#             new_K.as_matrix(),
#         )
#         if crop:
#             x, y, w, h = roi
#             undistorted = undistorted[y : y + h, x : x + w]
#         return image_processing.Image(undistorted, image.timestamp)


# @dataclass
# class OrthographicCamera:
#     T: SE3
