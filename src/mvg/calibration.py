#!/usr/bin/env python3
""" This module implements camera calibration process.

TODO: Implement tangential distortion calibration.
"""

import math
from enum import IntEnum
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mvg import basic, homography
from mvg.camera import Camera, CameraMatrix, RadialDistortionModel


def get_chessboard_object_points(*, rows: int, cols: int, grid_size: float):
    """Util functions for getting planar chess board object points"""
    object_points = np.zeros(shape=(cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.transpose(np.mgrid[:cols, :rows], (2, 1, 0)).reshape(-1, 2)
    object_points *= grid_size
    return object_points


def find_corners(*, image, grid_rows, grid_cols):
    """Detect chess board corners from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(gray, (grid_cols, grid_rows))
    if corners is None or len(corners) == 0:
        return
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # TODO: Check `cornerSubPix`
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)


def compute_reprejection_error(
    *,
    image_points: np.ndarray,
    object_points_W: np.ndarray,
    camera_matrix: CameraMatrix,
    T_CW: basic.SE3,
    radial_distortion_model: Optional[RadialDistortionModel] = None,
) -> float:
    """Compute reprojection error

    NOTE: Currently only support radial distortion model.
    """
    camera = Camera(K=camera_matrix, k=radial_distortion_model, T=T_CW)
    reprojected = camera.project_points(object_points_W)
    rms = math.sqrt((np.linalg.norm(image_points - reprojected, axis=1) ** 2).mean())
    return rms


class _ZhangsMethod:
    """Zhang's method for estimating camera parameters

    Reference:
        Zhengyou Zhang. A flexible new technique for camera calibration.
        Pattern Analysis and Machine Intelligence,
        IEEE Transactions on, 22(11):1330–1334, 2000.
    """

    @staticmethod
    def _get_homographies(image_points, object_points):
        homographies = []
        for points in image_points:
            H = homography.HomographySolver2d.compute(src=object_points[:, :2], dst=points)
            if H is not None:
                homographies.append(H.as_matrix())
        return homographies

    @staticmethod
    def _get_v(H, i, j):
        """Compute v vector from i-th and j-th columns of the homography H"""
        h_i1, h_i2, h_i3 = H[:, i]
        h_j1, h_j2, h_j3 = H[:, j]

        return np.asarray(
            [
                h_i1 * h_j1,
                h_i1 * h_j2 + h_i2 * h_j1,
                h_i2 * h_j2,
                h_i3 * h_j1 + h_i1 * h_j3,
                h_i3 * h_j2 + h_i2 * h_j3,
                h_i3 * h_j3,
            ]
        )

    @staticmethod
    def _get_intrinsics(homographies):
        V = []
        for i, H in enumerate(homographies):
            # Eq. 8
            V.append(
                [
                    _ZhangsMethod._get_v(H, 0, 1),
                    _ZhangsMethod._get_v(H, 0, 0) - _ZhangsMethod._get_v(H, 1, 1),
                ]
            )
        V = np.asarray(V).reshape(-1, 6)

        # Eq. 9
        _, _, vt = np.linalg.svd(V)
        b = vt[-1]

        #
        # Method 1
        #

        B = np.asarray(
            [
                [b[0], b[1], b[3]],
                [b[1], b[2], b[4]],
                [b[3], b[4], b[5]],
            ]
        )

        # B = K^(-T) @ K^(-1)
        # chol(B) = L @ L.T, where L = K^(-T)

        # B is symmetric, but not necessarily positive definite, since lambda might be negative.
        # Invert the sign when not all diagonal elements are positive.
        if not np.all(np.diag(B) > 0):
            B = -B
        L = np.linalg.cholesky(B)
        K = np.linalg.inv(L).T
        K /= K[-1, -1]

        #
        # Method 2
        #

        # Appendix B uses the method below, which is not clear how "without difficult" it is.
        # Cholesky decomposition is more straight forward.

        # denominator = b[0] * b[2] - b[1] * b[1]

        # u0 = (b[1] * b[4] - b[2] * b[3]) / denominator
        # v0 = (b[1] * b[3] - b[0] * b[4]) / denominator

        # lmda = (
        #     b[0] * b[2] * b[5]
        #     - b[1] * b[1] * b[5]
        #     - b[0] * b[4] * b[4]
        #     + 2 * b[1] * b[3] * b[4]
        #     - b[2] * b[3] * b[3]
        # )

        # alpha = math.sqrt(lmda / (denominator * b[0]))
        # beta = math.sqrt(lmda / denominator ** 2 * b[0])
        # gamma = math.sqrt(lmda / (denominator ** 2 * b[0])) * b[1]

        # K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

        return CameraMatrix.from_matrix(K)

    @staticmethod
    def _get_extrinsics(
        homographies: List[np.ndarray], camera_matrix: CameraMatrix
    ) -> List[basic.SE3]:
        all_poses = []
        # To match the notation in the paper, re-assign some variables
        A = camera_matrix.as_matrix()
        A_inv = np.linalg.inv(A)
        for H in homographies:
            h1 = H[:, 0].reshape(3, 1)
            h2 = H[:, 1].reshape(3, 1)
            h3 = H[:, 2].reshape(3, 1)
            lmda = 1.0 / np.linalg.norm(A_inv @ h1)
            # lmda = np.linalg.norm(A_inv @ h2) yields the same value
            r1 = lmda * A_inv @ h1
            r2 = lmda * A_inv @ h2
            r3 = np.cross(r1.T, r2.T).T  # Cross doesn't like column vectors
            t = lmda * A_inv @ h3
            R = np.hstack([r1, r2, r3])
            all_poses.append(basic.SE3(Rotation.from_matrix(R), t))
        return all_poses

    @staticmethod
    def _get_radial_distortion_coeffs(
        all_image_points: np.ndarray,
        object_points_W: np.ndarray,
        camera_matrix: CameraMatrix,
        all_extrinsics: List[basic.SE3],
    ) -> np.ndarray:

        cx = camera_matrix.cx
        cy = camera_matrix.cy

        # Eq. 13
        D = []
        d = []
        for i, image_points in enumerate(all_image_points):

            extrinsics = all_extrinsics[i]
            # xy is the ideal points on normalized image plane
            xy = basic.homogenize(object_points_W) @ extrinsics.as_augmented_matrix().T

            # uv is the ideal points in the pixel image plane
            uv = xy @ camera_matrix.as_matrix().T
            uv /= uv[:, -1].reshape(-1, 1)
            uv = uv[:, :2]

            u = uv[:, 0]
            v = uv[:, 1]

            xy /= xy[:, -1].reshape(-1, 1)
            xy = xy[:, :2]

            r2 = np.linalg.norm(xy, axis=1) ** 2
            r4 = r2**2

            D.append([(u - cx) * r2, (u - cx) * r4])
            D.append([(v - cy) * r2, (v - cy) * r4])

            d.append(image_points[:, 0] - u)
            d.append(image_points[:, 1] - v)

        D = np.asarray(D).reshape(-1, 2)
        d = np.asarray(d).reshape(-1, 1)

        k = np.linalg.inv(D.T @ D) @ D.T @ d

        return np.r_[k.T[0], 0.0]

    @staticmethod
    def _residual(
        param_array: np.ndarray,
        all_image_points: np.ndarray,
        object_points_W: np.ndarray,
    ):
        components = _ZhangsMethod._decompose_parameters(
            param_array=param_array, n_points=len(all_image_points)
        )
        camera_matrix = components[0]
        radial_distortion_model = components[1]
        all_extrinsics_array = components[2]
        residuals = []
        for i in range(len(all_image_points)):
            image_points = all_image_points[i]
            T_CW = basic.SE3.from_rotvec_pose(all_extrinsics_array[i])
            camera = Camera(K=camera_matrix, k=radial_distortion_model, T=T_CW)
            reprojected = camera.project_points(points_W=object_points_W)
            residuals.append(image_points - reprojected)
        residuals = np.asarray(residuals).reshape(-1)
        return residuals

    @staticmethod
    def _bundle_adjustment(
        *,
        camera_matrix: CameraMatrix,
        radial_distortion_model: RadialDistortionModel,
        all_extrinsics: List[basic.SE3],
        all_image_points: np.ndarray,
        object_points_W: np.ndarray,
    ) -> Tuple[CameraMatrix, RadialDistortionModel, List[basic.SE3]]:
        """Refine estimated camera intrinsic parameters"""

        x0 = _ZhangsMethod._compose_parameters(
            camera_matrix=camera_matrix,
            radial_distortion_model=radial_distortion_model,
            all_extrinsics=all_extrinsics,
        )

        result = least_squares(
            fun=_ZhangsMethod._residual,
            x0=x0,
            args=(all_image_points, object_points_W),
            loss="huber",
        )

        if not result["success"]:
            return (camera_matrix, radial_distortion_model, all_extrinsics)

        optimized_x = _ZhangsMethod._decompose_parameters(
            param_array=result["x"], n_points=len(all_image_points)
        )

        camera_matrix, radial_distortion_model, all_extrinsics_array = optimized_x
        all_extrinsics = [
            basic.SE3.from_rotvec_pose(extrinsics) for extrinsics in all_extrinsics_array
        ]
        return (camera_matrix, radial_distortion_model, all_extrinsics)

    @staticmethod
    def _compose_parameters(
        *,
        camera_matrix: CameraMatrix,
        radial_distortion_model: RadialDistortionModel,
        all_extrinsics: List[basic.SE3],
    ):
        """Compose parameters for calibration"""
        camera_param_array = camera_matrix.as_array()
        radial_distortion_model_array = radial_distortion_model.as_array()
        all_extrinsics_array = np.asarray([pose.as_rotvec_pose() for pose in all_extrinsics])
        all_extrinsics_array = all_extrinsics_array.reshape(-1)
        return np.r_[camera_param_array, radial_distortion_model_array, all_extrinsics_array]

    @staticmethod
    def _decompose_parameters(*, param_array: np.ndarray, n_points: int):
        """Decompose parameters from optimization to their original forms"""
        camera_matrix = CameraMatrix(*param_array[:5])
        radial_distortion_model = RadialDistortionModel(*param_array[5:8])
        all_extrinsics_array = param_array[8:].reshape(n_points, 6)
        return camera_matrix, radial_distortion_model, all_extrinsics_array

    @staticmethod
    def calibrate(all_image_points, object_points_W, debug=False):
        """Estimate camera intrinsic and radial distortion parameters"""
        assert len(all_image_points), "Not enough valid image points!"
        homographies = _ZhangsMethod._get_homographies(all_image_points, object_points_W)

        assert len(homographies) >= 3, "Not enough valid homographies!"
        camera_matrix = _ZhangsMethod._get_intrinsics(homographies)

        # Extrinsics are per image
        all_extrinsics = _ZhangsMethod._get_extrinsics(homographies, camera_matrix)

        distortion_coeffs = _ZhangsMethod._get_radial_distortion_coeffs(
            all_image_points=all_image_points,
            object_points_W=object_points_W,
            camera_matrix=camera_matrix,
            all_extrinsics=all_extrinsics,
        )

        radial_distortion_model = RadialDistortionModel(*distortion_coeffs)

        camera_matrix, radial_distortion_model, all_extrinsics = _ZhangsMethod._bundle_adjustment(
            camera_matrix=camera_matrix,
            radial_distortion_model=radial_distortion_model,
            all_extrinsics=all_extrinsics,
            all_image_points=all_image_points,
            object_points_W=object_points_W,
        )

        if debug:
            return (
                camera_matrix,
                radial_distortion_model,
                dict(homographies=homographies, all_extrinsics=all_extrinsics),
            )

        return camera_matrix, radial_distortion_model


class IntrinsicsCalibration:
    class Method(IntEnum):
        kZhangsMethod = 0

    @staticmethod
    def calibrate(
        image_points,
        object_points,
        method: Method = Method.kZhangsMethod,
        debug: bool = False,
    ):
        if method == IntrinsicsCalibration.Method.kZhangsMethod:
            return _ZhangsMethod.calibrate(image_points, object_points, debug)

        raise NotImplementedError(f"Not supported calibration method {method}!")


class StereoCalibration:
    @staticmethod
    def calibrate_from_calibrated_camera(
        image_points_l: np.ndarray,
        camera_matrix_l: CameraMatrix,
        radial_distortion_model_l: RadialDistortionModel,
        image_points_r: np.ndarray,
        camera_matrix_r: CameraMatrix,
        radial_distortion_model_r: RadialDistortionModel,
    ):
        pass
