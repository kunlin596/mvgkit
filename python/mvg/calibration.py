#!/usr/bin/env python3
import cv2
import tqdm
import numpy as np
import math

from typing import List, Optional
from scipy.spatial.transform import Rotation
from mvg import basic, homography, camera
from enum import IntEnum
from collections import namedtuple

CvIntrinsicsCalibrationData = namedtuple(
    "CvIntrinsicsCalibrationData",
    ["camera_matrix", "dist", "rvecs", "tvecs", "width", "height"],
)


def get_chessboard_object_points(*, rows, cols, grid_size):
    object_points = np.zeros(shape=(cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.transpose(np.mgrid[:cols, :rows], (2, 1, 0)).reshape(
        -1, 2
    )
    object_points *= grid_size
    return object_points


def find_corners(*, image, grid_rows, grid_cols):
    """Find chess board corners from color image"""
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
    object_points: np.ndarray,
    camera_matrix: camera.CameraMatrix,
    camera_pose: basic.SE3,
    radial_distortion_model: Optional[camera.RadialDistortionModel] = None,
) -> np.ndarray:
    object_points_C = object_points @ camera_pose.R.as_matrix().T + camera_pose.t

    if radial_distortion_model is not None:
        normalized_image_points = camera_matrix.project_to_normalized_image_plane(
            points_C=object_points_C
        )
        distorted_normalized_image_points = radial_distortion_model.distort(
            normalized_image_points=normalized_image_points
        )
        reprojected = camera_matrix.project_to_sensor_image_plane(
            normalized_image_points=distorted_normalized_image_points
        )
    else:
        reprojected = camera_matrix.project(object_points_C)

    rms = math.sqrt((np.linalg.norm(image_points - reprojected, axis=1) ** 2).mean())

    return rms


class _ZhangsMethod:
    @staticmethod
    def _get_homographies(image_points, object_points):
        homographies = []
        for points in image_points:
            H = homography.Homography.compute(src=object_points[:, :2], dst=points)
            if H is not None:
                homographies.append(H)
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

        return K

    @staticmethod
    def _get_extrinsics(homographies, intrinsics):
        all_poses = []
        # To match the notation in the paper, re-assign some variables
        A = intrinsics
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
        intrinsics: np.ndarray,
        all_extrinsics: List[basic.SE3],
    ) -> np.ndarray:
        camera_matrix = camera.CameraMatrix()
        camera_matrix.from_matrix(intrinsics)

        cx = camera_matrix.cx
        cy = camera_matrix.cy

        # Eq. 13
        D = []
        d = []
        for i, image_points in enumerate(all_image_points):
            extrinsics = all_extrinsics[i]

            # xy is the ideal points on normalized image plane
            xy = basic.homogeneous(object_points_W) @ extrinsics.as_augmented_matrix().T

            # uv is the ideal points in the pixel image plane
            uv = xy @ camera_matrix.as_matrix().T
            uv /= uv[:, -1].reshape(-1, 1)
            uv = uv[:, :2]

            u = uv[:, 0]
            v = uv[:, 1]

            xy /= xy[:, -1].reshape(-1, 1)
            xy = xy[:, :2]

            r2 = np.linalg.norm(xy, axis=1) ** 2
            r4 = r2 ** 2

            D.append([(u - cx) * r2, (u - cx) * r4])
            D.append([(v - cy) * r2, (v - cy) * r4])

            d.append(image_points[:, 0] - u)
            d.append(image_points[:, 1] - v)

        D = np.asarray(D).reshape(-1, 2)
        d = np.asarray(d).reshape(-1, 1)

        k = np.linalg.inv(D.T @ D) @ D.T @ d

        return np.r_[k.T[0], 0.0]

    @staticmethod
    def calibrate(all_image_points, object_points_W, debug=False):
        """
        Zhengyou Zhang. A flexible new technique for camera calibration.
        Pattern Analysis and Machine Intelligence,
        IEEE Transactions on, 22(11):1330–1334, 2000.
        """

        assert len(all_image_points), "Not enough valid image points!"
        homographies = _ZhangsMethod._get_homographies(
            all_image_points, object_points_W
        )

        assert len(homographies) >= 3, "Not enough valid homographies!"
        camera_matrix = _ZhangsMethod._get_intrinsics(homographies)

        # Extrinsics are per image
        all_extrinsics = _ZhangsMethod._get_extrinsics(homographies, camera_matrix)

        distortion_coeffs = _ZhangsMethod._get_radial_distortion_coeffs(
            all_image_points=all_image_points,
            object_points_W=object_points_W,
            intrinsics=camera_matrix,
            all_extrinsics=all_extrinsics,
        )

        radial_distortion_model = camera.RadialDistortionModel(*distortion_coeffs)

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


def intrinsic_calibration(*, images, grid_size=0.019):
    """Calibrate camera using images given calibration board grid size."""
    print(f"Detecting calibration pattern using {len(images):d} images...")
    object_points = np.zeros((6 * 8, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[:8, :6].T.reshape(-1, 2)
    # object_points *= grid_size

    all_image_points = []
    all_object_points = []
    for image in tqdm.tqdm(images):
        corners = find_corners(image)
        if corners is not None:
            all_object_points.append(object_points)
            all_image_points.append(corners)

    assert len(all_image_points) > 0, "Calibration pattern detection failed!"

    width = images[0].shape[1]
    height = images[0].shape[0]
    print(f"Calibrating camera, this might take a while...")
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, (width, height), None, None
    )
    print("Done calibration.")
    print("Get optimal camera matrix...")
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist, (width, height), 1, (width, height)
    )
    print("Undistorting images...")
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist, None, new_camera_matrix, (width, height), 5
    )

    # print("Undistort the images for verification...")
    # plt.ion()
    # x, y, w, h = roi
    # for index, image in enumerate(images):
    #     distorted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     undistorted = cv2.remap(distorted, mapx, mapy, cv2.INTER_LINEAR)
    #     undistorted = undistorted[y : y + h, x : x + w]
    #     plt.subplot(121)
    #     plt.title(f"Distorted {index:d}/{len(images):d}")
    #     plt.imshow(distorted)
    #     plt.subplot(122)
    #     plt.title(f"Undistorted {index:d}/{len(images):d}")
    #     plt.imshow(undistorted)
    #     plt.show()
    #     input()

    # print("Undistort the images for verification...")
    # plt.ion()
    # x, y, w, h = roi
    # for image in images:
    #     distorted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     undistorted = cv2.undistort(
    #         distorted,
    #         camera_matrix,
    #         dist,
    #         None,
    #         new_camera_matrix,
    #     )
    #     undistorted = undistorted[y : y + h, x : x + w]
    #     plt.subplot(121)
    #     plt.title("Distorted")
    #     plt.imshow(distorted)
    #     plt.subplot(122)
    #     plt.title("Undistorted")
    #     plt.imshow(undistorted)
    #     plt.show()
    #     input()

    # embed()
    return IntrisicsCalibrationData(
        camera_matrix=camera_matrix,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        width=width,
        height=height,
    )


def optimize_camera_matrix(*, calibration_data):
    width = calibration_data.width
    height = calibration_data.height
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        calibration_data.camera_matrix,
        calibration_data.dist,
        (width, height),
        1,
        (width, height),
    )
    return new_camera_matrix, roi


def stereo_calibration(
    *,
    images_left,
    calibration_data_left,
    images_right,
    calibration_data_right,
    pattern_grid,
    pattern_grid_size,
):

    height = pattern_grid[0]
    width = pattern_grid[1]
    object_points = np.zeros((height * width, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[:height, :width].T.reshape(-1, 2)
    object_points *= pattern_grid_size

    camera_matrix_left, roi_left = optimize_camera_matrix(
        calibration_data=calibration_data_left
    )
    camera_matrix_right, roi_right = optimize_camera_matrix(
        calibration_data=calibration_data_right
    )

    poses = []

    for image0, image1 in tqdm.tqdm(zip(images_left[::10], images_right[::10])):
        image0 = undistort_image(calibration_data_left, image0)
        image1 = undistort_image(calibration_data_right, image1)
        corners0 = find_corners(image0)
        corners1 = find_corners(image1)

        if corners0 is None or corners1 is None:
            continue

        retval_left, rvec_left, tvec_left = cv2.solvePnP(
            object_points,
            corners0,
            camera_matrix_left,
            None,
        )
        retval_right, rvec_right, tvec_right = cv2.solvePnP(
            object_points,
            corners1,
            camera_matrix_right,
            None,
        )

        if retval_left and retval_right:
            rot_left = Rotation.from_rotvec(rvec_left.reshape(3)).as_matrix()
            rot_right = Rotation.from_rotvec(rvec_right.reshape(3)).as_matrix()

            T0 = basic.homogeneous_transformation(rot_left, tvec_left.reshape(3))
            T1 = basic.homogeneous_transformation(rot_right, tvec_right.reshape(3))

            poses.append(T0 @ T1.T)

    rvecs = []
    tvecs = []
    for pose in poses:
        rvecs.append(Rotation.from_matrix(pose[:3, :3]).as_rotvec())
        tvecs.append(pose[:3, 3])
    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)

    print(
        f"Rotation std: {np.linalg.norm(rvecs, axis=1).std(axis=0):7.3f}(rad), translation std: {np.linalg.norm(tvecs, axis=1).std():7.3f}(m)"
    )

    # fig = plt.figure(0)
    # ax1 = fig.add_subplot(121, projection="3d")
    # ax1.plot(rvecs[:, 0], rvecs[:, 1], rvecs[:, 2], linewidth=0, marker="o", color="r")
    # ax1.set_title("Rotation Vectors")

    # ax1 = fig.add_subplot(122, projection="3d")
    # ax1.plot(tvecs[:, 0], tvecs[:, 1], tvecs[:, 2], linewidth=0, marker="o", color="r")
    # ax1.set_title("Translation Vectors")

    # plt.legend()
    # plt.ion()
    # plt.show()

    # embed()

    return Rotation.from_rotvec(rvecs.mean(axis=0)).as_matrix(), tvecs.mean(axis=0)


def undistort_image(calibration_data, image):
    camera_matrix = calibration_data.camera_matrix
    dist = calibration_data.dist
    width = calibration_data.width
    height = calibration_data.height
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist, (width, height), 1, (width, height)
    )
    x, y, w, h = roi
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist, None, new_camera_matrix, (width, height), 5
    )
    undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    undistorted = undistorted[y : y + h, x : x + w]
    return undistorted
