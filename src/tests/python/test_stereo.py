#!/usr/bin/env python3
from math import sqrt

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from _mvgkit_geometry_cppimpl import intersect_lines_2d
from mvgkit import stereo
from mvgkit.homography import Homography2d
from mvgkit.log import logger
from mvgkit.stereo import (
    AffinityRecoverySolver,
    Essential,
    EssentialOptions,
    Fundamental,
    FundamentalOptions,
    StereoMatcher,
    StereoRectifier,
)

np.set_printoptions(suppress=True, precision=7, linewidth=120)


#
# Test cases
#


def test_estimation_manual_data_association(leuven_stereo_data_pack):
    """Test the convergence of intersections of reprojected epilines on the other view
    using manually associated points."""

    points_L = leuven_stereo_data_pack.manual_points_L
    points_R = leuven_stereo_data_pack.manual_points_R
    options = FundamentalOptions()
    options.atol = 2.8  # NOTE(kun): manual annotation error is large by me.
    fundamental = Fundamental(options, points_L, points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = fundamental.get_inlier_indices()
    lines_L = stereo.get_epilines(points_R[inlier_indices], F_RL)
    lines_R = stereo.get_epilines(points_L[inlier_indices], F_RL.T)

    np.testing.assert_allclose(stereo.get_epipole(F_RL), intersect_lines_2d(lines_L), atol=0.2)
    np.testing.assert_allclose(stereo.get_epipole(F_RL.T), intersect_lines_2d(lines_R), atol=0.2)


def test_estimation_with_detected_keypoints(leuven_stereo_data_pack):
    """Test the convergence of intersections of reprojected epilines on the other view
    using detected SIFT keypoints."""

    points_L = leuven_stereo_data_pack.points_L
    points_R = leuven_stereo_data_pack.points_R
    fundamental = Fundamental(FundamentalOptions(), points_L, points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = fundamental.get_inlier_indices()
    lines_L = stereo.get_epilines(points_R[inlier_indices], F_RL)
    lines_R = stereo.get_epilines(points_L[inlier_indices], F_RL.T)

    np.testing.assert_allclose(stereo.get_epipole(F_RL), intersect_lines_2d(lines_L), atol=0.2)
    np.testing.assert_allclose(stereo.get_epipole(F_RL.T), intersect_lines_2d(lines_R), atol=0.2)


def test_two_view_reprojection_error(leuven_stereo_data_pack, stereo_reprojection_rms_threshold):
    camera_matrix = leuven_stereo_data_pack.camera_matrix
    points_L = leuven_stereo_data_pack.manual_points_L
    points_R = leuven_stereo_data_pack.manual_points_R

    essential = Essential(EssentialOptions(), points_L, points_R, camera_matrix)
    inlier_indices = essential.get_inlier_indices()
    points_inliers_L = points_L[inlier_indices]
    points_inliers_R = points_R[inlier_indices]

    points3d_L = essential.get_points3d_L()
    rmat_RL, t_RL = essential.get_pose_RL()
    R_RL = Rotation.from_matrix(rmat_RL)

    camera_matrix = leuven_stereo_data_pack.camera_matrix
    reprojected_L = camera_matrix.project(points3d_L)
    reprojected_R = camera_matrix.project(R_RL.apply(points3d_L) + t_RL)

    reprojected_diff_L = reprojected_L - points_inliers_L
    reprojected_diff_R = reprojected_R - points_inliers_R
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
