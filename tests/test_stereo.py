#!/usr/bin/env python3

import json
from math import sqrt
from mvg import camera
from pytest import fixture
import cv2

import numpy as np

from pathlib import Path
from dataclasses import dataclass

from mvg import stereo
from mvg.basic import SE3
from mvg.camera import CameraMatrix

from mvg import features
from mvg.stereo import Fundamental
from mvg.stereo import decompose_essential_matrix

np.set_printoptions(suppress=True, precision=7, linewidth=120)


@fixture
def points_L():
    return np.array(
        [
            [75, 297],
            [101, 304],
            [98, 386],
            [70, 386],
            [115, 18],
            [93, 24],
            [101, 45],
            [366, 164],
            [392, 173],
            [566, 269],
            [522, 62],
        ],
        dtype=np.float32,
    )


@fixture
def points_R():
    return np.array(
        [
            [366, 330],
            [381, 333],
            [383, 384],
            [368, 384],
            [383, 124],
            [369, 157],
            [376, 172],
            [592, 197],
            [621, 200],
            [656, 310],
            [734, 91],
        ],
        dtype=np.float32,
    )


@dataclass
class StereoDataPack:
    image_L: np.ndarray
    image_R: np.ndarray
    points_L: np.ndarray
    points_R: np.ndarray
    camera_matrix: CameraMatrix


@fixture
def stereo_data_pack(data_root_path, points_L, points_R):
    fundamental_root_path = Path(data_root_path) / "fundamental"
    with open(fundamental_root_path / "meta.json", "r") as f:
        meta = json.load(f)
    image_L = cv2.imread(str(fundamental_root_path / meta["left"]))
    image_L = cv2.cvtColor(image_L, cv2.COLOR_BGR2RGB)
    image_R = cv2.imread(str(fundamental_root_path / meta["right"]))
    image_R = cv2.cvtColor(image_R, cv2.COLOR_BGR2RGB)
    camera_matrix = CameraMatrix.from_matrix(np.reshape(meta["K"], (3, 3)))

    return StereoDataPack(
        image_L=image_L,
        image_R=image_R,
        points_L=points_L,
        points_R=points_R,
        camera_matrix=camera_matrix,
    )


def test_fundamental_matrix_manual_correspondence(
    stereo_data_pack: StereoDataPack, fundamental_rms_threshold: float
):
    # print("Reading data...")
    # fundamental_root_path = Path(data_root_path) / "fundamental"
    # with open(fundamental_root_path / "meta.json", "r") as f:
    #     meta = json.load(f)

    # image_L = cv2.imread(str(fundamental_root_path / meta["left"]))
    # image_R = cv2.imread(str(fundamental_root_path / meta["right"]))
    # camera_matrix = CameraMatrix.from_matrix(np.reshape(meta["K"], (3, 3)))

    # TODO(kun): after implementing RANSAC point registration, enable auto matching again
    # print("Computing feature points and their matches on left and right images...")
    # keypoints_L, descriptors_L = features.SIFT.detect(image_L)
    # keypoints_R, descriptors_R = features.SIFT.detect(image_R)
    # matches = features.Matcher.match(descriptors1=descriptors_L, descriptors2=descriptors_R)
    # points_L, points_R, _ = features.Matcher.get_matched_points(
    #     keypoints_L, keypoints_R, matches, dist_threshold=0.3
    # )

    points_L = stereo_data_pack.points_L
    points_R = stereo_data_pack.points_R

    print("Computing fundamental matrix...")
    F_RL, inlier_mask = Fundamental.compute(x_L=points_L, x_R=points_R)
    rms = Fundamental.compute_geometric_rms(
        F_RL=F_RL, x_L=points_L[inlier_mask], x_R=points_R[inlier_mask]
    )

    print("\nComputing OpenCV fundamental matrix for performance reference...")

    Fcv_RL, _ = cv2.findFundamentalMat(points_L, points_R)
    Fcv_LR = Fcv_RL.T

    rms_cv = Fundamental.compute_geometric_rms(F_RL=Fcv_LR, x_L=points_L, x_R=points_R)

    print(
        "".join(
            [
                f"rms={rms:7.3f}, opencv_rms={rms_cv:7.3f}, ",
                f"{'Won' if rms < rms_cv else 'Lost':5s}, ",
                f"F-norm={np.linalg.norm((F_RL - Fcv_RL)):7.3f}",
            ]
        )
    )

    assert rms < fundamental_rms_threshold  # in pixel


def _get_R_t(R1_RL, R2_RL, t_R, K, points_L, points_R):
    P1 = K @ SE3.from_rotmat_tvec(np.eye(3), np.zeros(3)).as_augmented_matrix()

    T_RL_candidates = [
        SE3.from_rotmat_tvec(R1_RL, t_R),
        SE3.from_rotmat_tvec(R1_RL, -t_R),
        SE3.from_rotmat_tvec(R2_RL, t_R),
        SE3.from_rotmat_tvec(R2_RL, -t_R),
    ]

    P2_candidates = [K @ T_RL.as_augmented_matrix() for T_RL in T_RL_candidates]

    max_num_valid_points = -1

    best_T = None
    best_points_3d = None
    best_inlier_mask = None
    all_inlier_masks = []

    for i, P2 in enumerate(P2_candidates):
        points_3d = stereo.triangulate(P1, P2, points_L, points_R)

        inlier_mask = points_3d[:, 2] > 1.0
        num_valid_points = np.count_nonzero(inlier_mask)

        all_inlier_masks.append(inlier_mask)

        if num_valid_points > max_num_valid_points:
            max_num_valid_points = num_valid_points

            best_T = T_RL_candidates[i]
            best_points_3d = points_3d
            best_inlier_mask = inlier_mask

    return best_T, best_points_3d, best_inlier_mask, T_RL_candidates, all_inlier_masks


def test_two_view_reprojection_error(
    stereo_data_pack: StereoDataPack, stereo_reprojection_rms_threshold: float
):
    camera_matrix = stereo_data_pack.camera_matrix
    image_L = stereo_data_pack.image_L
    image_R = stereo_data_pack.image_R

    # TODO(kun): after implementing RANSAC point registration, enable auto matching again
    print("Computing feature points and their matches on left and right images...")
    keypoints_L, descriptors_L = features.SIFT.detect(image_L)
    keypoints_R, descriptors_R = features.SIFT.detect(image_R)
    matches = features.Matcher.match(descriptors1=descriptors_L, descriptors2=descriptors_R)
    points_L, points_R, _ = features.Matcher.get_matched_points(
        keypoints_L, keypoints_R, matches, dist_threshold=0.8
    )

    F_RL, inlier_mask = Fundamental.compute(x_L=points_L, x_R=points_R)
    points_inliers_L = points_L[inlier_mask]
    points_inliers_R = points_R[inlier_mask]

    K = camera_matrix.as_matrix()
    E_RL = K.T @ F_RL @ K

    R1_RL, R2_RL, t_R = decompose_essential_matrix(E_RL=E_RL)
    T_RL, points_3d, points_3d_inlier_mask, T_RL_candidates, all_inlier_masks = _get_R_t(
        R1_RL, R2_RL, t_R, K, points_inliers_L, points_inliers_R
    )

    reprojected_L = camera.project_points(
        object_points_W=points_3d[points_3d_inlier_mask], camera_matrix=camera_matrix
    )
    reprojected_R = camera.project_points(
        object_points_W=points_3d[points_3d_inlier_mask], camera_matrix=camera_matrix, T_CW=T_RL
    )

    reprojected_diff_L = reprojected_L - points_inliers_L[points_3d_inlier_mask]
    reprojected_diff_R = reprojected_R - points_inliers_R[points_3d_inlier_mask]
    rms_L = sqrt((np.linalg.norm(reprojected_diff_L, axis=1) ** 2).mean())
    rms_R = sqrt((np.linalg.norm(reprojected_diff_R, axis=1) ** 2).mean())

    print(f"rms_L: {rms_L:7.3f}")
    print(f"rms_R: {rms_R:7.3f}")

    # import matplotlib.pyplot as plt

    # width = image_L.shape[1]
    # height = image_L.shape[0]

    # plt.figure(figsize=(32, 24))
    # for i, T_RL in enumerate(T_RL_candidates):
    #     print(f"Visualizing {i}-th candidate.")
    #     points_3d_inlier_mask = all_inlier_masks[i]

    #     reprojected_L = camera.project_points(
    #         object_points_W=points_3d[points_3d_inlier_mask], camera_matrix=camera_matrix
    #     )
    #     reprojected_R = camera.project_points(
    #         object_points_W=points_3d[points_3d_inlier_mask], camera_matrix=camera_matrix, T_CW=T_RL
    #     )

    #     reprojected_diff_L = reprojected_L - points_inliers_L[points_3d_inlier_mask]
    #     reprojected_diff_R = reprojected_R - points_inliers_R[points_3d_inlier_mask]
    #     rms_L = sqrt((np.linalg.norm(reprojected_diff_L, axis=1) ** 2).mean())
    #     rms_R = sqrt((np.linalg.norm(reprojected_diff_R, axis=1) ** 2).mean())

    #     print(f"rms_L: {rms_L:7.3f}")
    #     print(f"rms_R: {rms_R:7.3f}")
    #     plt.subplot(2, 4, i * 2 + 1)
    #     plt.title(f"Projected points in L, rms: {rms_L:7.3f} px")
    #     plt.imshow(image_L)
    #     plt.xlim([0, width])
    #     plt.ylim([height, 0])
    #     plt.scatter(points_inliers_L[:, 0], points_inliers_L[:, 1], alpha=0.5, label="Input", c="r")
    #     plt.scatter(reprojected_L[:, 0], reprojected_L[:, 1], alpha=0.5, label="Reprojected", c="g")
    #     plt.legend()

    #     plt.subplot(2, 4, i * 2 + 2)
    #     plt.title(f"Projected points in R, rms: {rms_R:7.3f} px")
    #     plt.imshow(image_R)
    #     plt.xlim([0, width])
    #     plt.ylim([height, 0])
    #     plt.scatter(points_inliers_R[:, 0], points_inliers_R[:, 1], alpha=0.5, label="Input", c="r")
    #     plt.scatter(reprojected_R[:, 0], reprojected_R[:, 1], alpha=0.5, label="Reprojected", c="g")
    #     plt.legend()

    # plt.tight_layout()
    # plt.show()

    # Fundamental.plot_epipolar_lines(image_L, image_R, points_inliers_L, points_inliers_R, F_RL)

    threshold = stereo_reprojection_rms_threshold
    assert rms_L < threshold, f"rms_L: {rms_L:7.3f} > {threshold}"
    assert rms_R < threshold, f"rms_R: {rms_R:7.3f} > {threshold}"
