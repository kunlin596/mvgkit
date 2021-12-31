#!/usr/bin/env python3

from math import sqrt
import cv2

import numpy as np

from mvg.basic import SE3, homogeneous
from mvg.camera import project_points
from mvg.stereo import AffinityRecoverySolver, Fundamental, decompose_essential_matrix, triangulate

from stereo_data_fixtures import StereoDataPack


np.set_printoptions(suppress=True, precision=7, linewidth=120)


def _solve_line_intersections(lines):
    A = lines[:, :2]
    b = -lines[:, 2]
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    return x


def test_fundamental_matrix_manual_correspondence(
    leuven_stereo_data_pack: StereoDataPack, fundamental_rms_threshold: float
):
    points_L = leuven_stereo_data_pack.points_L
    points_R = leuven_stereo_data_pack.points_R

    print("Computing fundamental matrix...")
    F_RL, inlier_mask = Fundamental.compute(x_L=points_L, x_R=points_R)
    rms = Fundamental.compute_geometric_rms(
        F_RL=F_RL, x_L=points_L[inlier_mask], x_R=points_R[inlier_mask]
    )

    print("\nComputing OpenCV fundamental matrix for performance reference...")

    Fcv_RL, _ = cv2.findFundamentalMat(points_L, points_R)
    Fcv_LR = Fcv_RL.T

    rms_cv = Fundamental.compute_geometric_rms(F_RL=Fcv_LR, x_L=points_L, x_R=points_R)

    lines_L = Fundamental.get_epilines_L(x_R=homogeneous(points_R), F_RL=F_RL)
    assert np.allclose(
        Fundamental.get_epipole_L(F_RL=F_RL)[:2], _solve_line_intersections(lines_L)
    ), "Left epipoles from F is not the same as the intersection of all left epilines!"

    lines_R = Fundamental.get_epilines_R(x_L=homogeneous(points_L), F_RL=F_RL)
    assert np.allclose(
        Fundamental.get_epipole_R(F_RL=F_RL)[:2], _solve_line_intersections(lines_R)
    ), "Right epipoles from F is not the same as the intersection of all right epilines!"

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
        points_3d = triangulate(P1, P2, points_L, points_R)

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
    leuven_stereo_data_pack: StereoDataPack, stereo_reprojection_rms_threshold: float
):
    camera_matrix = leuven_stereo_data_pack.camera_matrix
    # image_L = leuven_stereo_data_pack.image_L
    # image_R = leuven_stereo_data_pack.image_R
    points_L = leuven_stereo_data_pack.points_L
    points_R = leuven_stereo_data_pack.points_R

    F_RL, inlier_mask = Fundamental.compute(x_L=points_L, x_R=points_R)
    points_inliers_L = points_L[inlier_mask]
    points_inliers_R = points_R[inlier_mask]

    K = camera_matrix.as_matrix()
    E_RL = K.T @ F_RL @ K

    camera_matrix = leuven_stereo_data_pack.camera_matrix
    R1_RL, R2_RL, t_R = decompose_essential_matrix(E_RL=E_RL)
    T_RL, points_3d, points_3d_inlier_mask, T_RL_candidates, all_inlier_masks = _get_R_t(
        R1_RL, R2_RL, t_R, K, points_inliers_L, points_inliers_R
    )

    reprojected_L = project_points(
        object_points_W=points_3d[points_3d_inlier_mask], camera_matrix=camera_matrix
    )
    reprojected_R = project_points(
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

    #     reprojected_L = project_points(
    #         object_points_W=points_3d[points_3d_inlier_mask], camera_matrix=camera_matrix
    #     )
    #     reprojected_R = project_points(
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


def test_stereo_rectification(book_stereo_data_pack: StereoDataPack):
    image_L = book_stereo_data_pack.image_L
    image_R = book_stereo_data_pack.image_R
    points_L = book_stereo_data_pack.points_L
    points_R = book_stereo_data_pack.points_R

    F_RL = book_stereo_data_pack.F_RL
    inlier_mask = book_stereo_data_pack.inlier_mask

    points_inliers_L = points_L[inlier_mask]
    points_inliers_R = points_R[inlier_mask]

    H_L, H_R = AffinityRecoverySolver.solve(
        F_RL=F_RL, image_shape_L=image_L.shape, image_shape_R=image_R.shape
    )

    lines_L = Fundamental.get_epilines_L(x_R=homogeneous(points_inliers_R), F_RL=F_RL)
    warped_lines_L = lines_L @ np.linalg.inv(H_L)
    assert np.allclose(
        (warped_lines_L[:, 1] / warped_lines_L[:, 0]).ptp(), 0.0
    ), "Warped lines in left image are not parallel to each other!"

    lines_R = Fundamental.get_epilines_R(x_L=homogeneous(points_inliers_L), F_RL=F_RL)
    warped_lines_R = lines_R @ np.linalg.inv(H_R)
    assert np.allclose(
        (warped_lines_R[:, 1] / warped_lines_R[:, 0]).ptp(), 0.0
    ), "Wrapped lines in right image are not parallel to each other!"
