#!/usr/bin/env python3

import json
from pytest import fixture
import cv2

import numpy as np

from pathlib import Path

from mvg import stereo
from mvg.basic import SE3
from mvg.camera import CameraMatrix

# from mvg import features
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


def test_fundamental_matrix_manual_correspondence(
    data_root_path,
    points_L,
    points_R,
    fundamental_rms_threshold: float,
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

    print("Computing fundamental matrix...")
    F_LR = Fundamental.compute(x_L=points_L, x_R=points_R)
    rms = Fundamental.compute_geometric_rms(F_LR=F_LR, x_L=points_L, x_R=points_R)

    print("\nComputing OpenCV fundamental matrix for performance reference...")

    Fcv_RL, _ = cv2.findFundamentalMat(points_L, points_R)
    Fcv_LR = Fcv_RL.T

    rms_cv = Fundamental.compute_geometric_rms(F_LR=Fcv_LR, x_L=points_L, x_R=points_R)

    print(
        "".join(
            [
                f"rms={rms:7.3f}, opencv_rms={rms_cv:7.3f}, ",
                f"{'Won' if rms < rms_cv else 'Lost':5s}, ",
                f"F-norm={np.linalg.norm((F_LR - Fcv_RL)):7.3f}",
            ]
        )
    )

    assert rms < fundamental_rms_threshold  # in pixel


def test_essential_matrix_decomposition(data_root_path, points_L, points_R):
    print("Reading data...")
    fundamental_root_path = Path(data_root_path) / "fundamental"
    with open(fundamental_root_path / "meta.json", "r") as f:
        meta = json.load(f)

    camera_matrix = CameraMatrix.from_matrix(np.reshape(meta["K"], (3, 3)))

    F_LR = Fundamental.compute(x_L=points_L, x_R=points_R)

    K = camera_matrix.as_matrix()
    E_LR = K.T @ F_LR @ K
    # Ecv_LR = K.T @ Fcv_LR @ K

    R1_LR, R2_LR, t_LR = decompose_essential_matrix(E_LR=E_LR)
    # R1cv_LR, R2cv_LR, tcv_LR = cv2.decomposeEssentialMat(Ecv_LR)

    P1 = K @ SE3.from_rotmat_tvec(np.eye(3), np.zeros(3)).as_augmented_matrix()
    P2_candidates = [
        K @ SE3.from_rotmat_tvec(R1_LR, t_LR).as_augmented_matrix(),
        K @ SE3.from_rotmat_tvec(R1_LR, -t_LR).as_augmented_matrix(),
        K @ SE3.from_rotmat_tvec(R2_LR, t_LR).as_augmented_matrix(),
        K @ SE3.from_rotmat_tvec(R2_LR, -t_LR).as_augmented_matrix(),
    ]

    points_3d = None
    for P2 in P2_candidates:
        points_3d_candidates = np.asarray(
            [stereo.triangulate(P1, P2, p1, p2) for p1, p2 in zip(points_L, points_R)]
        )
        valid_points_3d = points_3d_candidates[points_3d_candidates[:, 2] > 0]
        if len(valid_points_3d) == len(points_3d_candidates):
            points_3d = points_3d_candidates

    # TODO(kun): Add precision test
    assert points_3d is not None

    # import matplotlib.pyplot as plt

    # plt.ion()
    # Fundamental.plot_epipolar_lines(
    #     image_L=image_L, image_R=image_R, points_L=points_L, points_R=points_R, F_LR=Fcv_LR
    # )

    # Fundamental.plot_epipolar_lines(
    #     image_L=image_L, image_R=image_R, points_L=points_L, points_R=points_R, F_LR=F_LR
    # )
