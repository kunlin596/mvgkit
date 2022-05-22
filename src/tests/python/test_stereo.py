#!/usr/bin/env python3
from math import sqrt

import numpy as np
import pytest

from _mvgkit_geometry_cppimpl import intersect_lines_2d
from mvgkit import stereo
from mvgkit.basic import SE3
from mvgkit.camera import Camera
from mvgkit.homography import Homography2d
from mvgkit.log import logger
from mvgkit.stereo import (
    AffinityRecoverySolver,
    Fundamental,
    FundamentalOptions,
    StereoMatcher,
    StereoRectifier,
    decompose_essential_matrix,
    triangulate,
)

np.set_printoptions(suppress=True, precision=7, linewidth=120)


def _get_R_t(R1_RL, R2_RL, t_R, K, points_L, points_R):
    # TODO: Replace this function
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


#
# Test cases
#


def _compute_geometric_rms(F_RL, x_L, x_R):
    residuals_L = stereo.compute_reprojection_residuals(F_RL, x_L, x_R)
    residuals_R = stereo.compute_reprojection_residuals(F_RL.T, x_R, x_L)
    np.testing.assert_array_less(residuals_L.std(), 1.0)
    np.testing.assert_array_less(residuals_R.std(), 1.0)
    return np.sqrt(np.r_[residuals_L**2, residuals_R**2]).mean()


def test_estimation_manual_data_association(leuven_stereo_data_pack):
    """Test the convergence of intersections of reprojected epilines on the other view
    using manually associated points."""

    points_L = leuven_stereo_data_pack.manual_points_L
    points_R = leuven_stereo_data_pack.manual_points_R
    options = FundamentalOptions()
    options.atol = 2.8  # NOTE(kun): manual annotation error is large by me.
    fundamental = Fundamental(options)
    fundamental(x_L=points_L, x_R=points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = list(fundamental.get_inlier_indices())
    lines_L = stereo.get_epilines(points_R[inlier_indices], F_RL)
    lines_R = stereo.get_epilines(points_L[inlier_indices], F_RL.T)

    np.testing.assert_allclose(stereo.get_epipole(F_RL), intersect_lines_2d(lines_L), atol=0.2)
    np.testing.assert_allclose(stereo.get_epipole(F_RL.T), intersect_lines_2d(lines_R), atol=0.2)


def test_estimation_with_detected_keypoints(leuven_stereo_data_pack):
    """Test the convergence of intersections of reprojected epilines on the other view
    using detected SIFT keypoints."""

    points_L = leuven_stereo_data_pack.points_L
    points_R = leuven_stereo_data_pack.points_R
    options = FundamentalOptions()
    fundamental = Fundamental(options)
    fundamental(x_L=points_L, x_R=points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = list(fundamental.get_inlier_indices())
    lines_L = stereo.get_epilines(points_R[inlier_indices], F_RL)
    lines_R = stereo.get_epilines(points_L[inlier_indices], F_RL.T)

    np.testing.assert_allclose(stereo.get_epipole(F_RL), intersect_lines_2d(lines_L), atol=0.2)
    np.testing.assert_allclose(stereo.get_epipole(F_RL.T), intersect_lines_2d(lines_R), atol=0.2)


@pytest.mark.skip(reason="Fix this test when essential matrix related features are re-implemented.")
def test_two_view_reprojection_error(leuven_stereo_data_pack, stereo_reprojection_rms_threshold):
    camera_matrix = leuven_stereo_data_pack.camera_matrix
    # image_L = leuven_stereo_data_pack.image_L
    # image_R = leuven_stereo_data_pack.image_R
    points_L = leuven_stereo_data_pack.points_L
    points_R = leuven_stereo_data_pack.points_R

    fundamental = Fundamental()
    fundamental(x_L=points_L, x_R=points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = fundamental.get_inlier_indices()
    points_inliers_L = points_L[inlier_indices]
    points_inliers_R = points_R[inlier_indices]

    K = camera_matrix.as_matrix()
    E_RL = K.T @ F_RL @ K

    camera_matrix = leuven_stereo_data_pack.camera_matrix
    R1_RL, R2_RL, t_R = decompose_essential_matrix(E_RL=E_RL)
    T_RL, points_3d, points_3d_inlier_mask, T_RL_candidates, all_inlier_masks = _get_R_t(
        R1_RL, R2_RL, t_R, K, points_inliers_L, points_inliers_R
    )
    camera = Camera(camera_matrix)
    reprojected_L = camera.project_points(points_3d[points_3d_inlier_mask])

    camera = Camera(camera_matrix, quat_CW=T_RL.R.as_quat(), trans_CW=T_RL.t)
    reprojected_R = camera.project_points(points_3d[points_3d_inlier_mask])

    reprojected_diff_L = reprojected_L - points_inliers_L[points_3d_inlier_mask]
    reprojected_diff_R = reprojected_R - points_inliers_R[points_3d_inlier_mask]
    rms_L = sqrt((np.linalg.norm(reprojected_diff_L, axis=1) ** 2).mean())
    rms_R = sqrt((np.linalg.norm(reprojected_diff_R, axis=1) ** 2).mean())

    logger.info(f"rms_L: {rms_L:7.3f}")
    logger.info(f"rms_R: {rms_R:7.3f}")

    threshold = stereo_reprojection_rms_threshold
    assert rms_L < threshold, f"rms_L: {rms_L:7.3f} > {threshold}"
    assert rms_R < threshold, f"rms_R: {rms_R:7.3f} > {threshold}"


@pytest.mark.skip(
    reason="Fix this test when point cloud generation related features are re-implemented."
)
def test_affinity_recovery(book_stereo_data_pack):
    image_L = book_stereo_data_pack.image_L
    image_R = book_stereo_data_pack.image_R
    points_L = book_stereo_data_pack.points_L
    points_R = book_stereo_data_pack.points_R

    F_RL = book_stereo_data_pack.F_RL
    inlier_indices = book_stereo_data_pack.inlier_indices

    points_inliers_L = points_L[inlier_indices]
    points_inliers_R = points_R[inlier_indices]

    H_L, H_R = AffinityRecoverySolver.solve(
        F_RL=F_RL, image_shape_L=image_L.shape, image_shape_R=image_R.shape
    )

    lines_L = stereo.get_epilines(x_R=points_inliers_R, F_RL=F_RL)
    warped_lines_L = lines_L @ np.linalg.inv(H_L)
    np.testing.assert_allclose((warped_lines_L[:, 1] / warped_lines_L[:, 0]).ptp(), 0.0)

    lines_R = stereo.get_epilines(x_L=points_inliers_L, F_RL=F_RL.T)
    warped_lines_R = lines_R @ np.linalg.inv(H_R)
    np.testing.allclose((warped_lines_R[:, 1] / warped_lines_R[:, 0]).ptp(), 0.0)


@pytest.mark.skip(
    reason="Fix this test when point cloud generation related features are re-implemented."
)
def test_stereo_rectification(book_stereo_data_pack):
    image_L = book_stereo_data_pack.image_L
    image_R = book_stereo_data_pack.image_R
    points_L = book_stereo_data_pack.points_L
    points_R = book_stereo_data_pack.points_R

    F_RL = book_stereo_data_pack.F_RL
    inlier_indices = book_stereo_data_pack.inlier_indices

    points_inliers_L = points_L[inlier_indices]
    points_inliers_R = points_R[inlier_indices]

    (
        H_L,
        H_R,
        _,  # size_L,
        _,  # size_R,
        _,  # rectified_image_corners_L,
        _,  # rectified_image_corners_R,
    ) = StereoRectifier.compute_rectification_homography(image_L, image_R, F_RL)

    rectified_points_L = Homography2d.from_matrix(H_L).transform(points_inliers_L)
    rectified_points_R = Homography2d.from_matrix(H_R).transform(points_inliers_R)

    y_diff_std = (rectified_points_L[:, 1] - rectified_points_R[:, 1]).std()
    threshold = 0.3
    assert (
        y_diff_std < threshold
    ), f"The std of y difference of rectified corresponding points is bigger than {threshold:7.3f} px!"


@pytest.mark.skip(
    reason="Fix this test when point cloud generation related features are re-implemented."
)
def test_stereo_disparity(aloe_stereo_data_pack):
    image_L = aloe_stereo_data_pack.image_L
    image_R = aloe_stereo_data_pack.image_R
    # points_L = aloe_stereo_data_pack.points_L
    # points_R = aloe_stereo_data_pack.points_R

    F_RL = aloe_stereo_data_pack.F_RL
    # inlier_mask = aloe_stereo_data_pack.inlier_mask

    matcher = StereoMatcher(F_RL, image_L, image_R)
    disparity_map = matcher.compute(image_L, image_R)

    # TODO(kun): Add more tests
    assert disparity_map is not None
