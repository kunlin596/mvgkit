#!/usr/bin/env python3
from math import sqrt

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mvgkit.common.log import logger
from mvgkit.common.utils import SkewSymmetricMatrix3d, homogenize
from mvgkit.estimation.stereo import (
    AffinityRecoverySolver,
    Homography2d,
    StereoMatcher,
    StereoRectifier,
)
from pymvgkit_common import CameraMatrix, intersect_lines_2d
from pymvgkit_estimation import (
    Essential,
    Fundamental,
    FundamentalOptions,
    get_epilines,
    get_epipole,
    get_homo_epipole,
)

np.set_printoptions(suppress=True, precision=15, linewidth=120)


@pytest.fixture
def fx():
    return 520.0


@pytest.fixture
def fy():
    return 520.0


@pytest.fixture
def cx():
    return 325.0


@pytest.fixture
def cy():
    return 250.0


@pytest.fixture
def s():
    return 0.0


@pytest.fixture
def camera_matrix(fx, fy, cx, cy, s):
    return CameraMatrix(fx, fy, cx, cy, s)


@pytest.fixture
def R_RL():
    return Rotation.from_euler("y", 60.0, degrees=True)


@pytest.fixture
def t_RL(R_RL):
    return -R_RL.apply([sqrt(3.0) / 2.0, 0.0, 0.5])


@pytest.fixture
def tx_RL(t_RL):
    return SkewSymmetricMatrix3d(t_RL)


@pytest.fixture
def E_RL(R_RL: Rotation, tx_RL: SkewSymmetricMatrix3d):
    R_RL = R_RL.as_matrix()
    tx_RL = tx_RL.as_matrix()
    E_RL = R_RL.T @ tx_RL.T
    return E_RL


@pytest.fixture
def F_RL(camera_matrix: CameraMatrix, E_RL: np.ndarray):
    Kinv = np.linalg.inv(camera_matrix.as_matrix())
    F_RL = Kinv.T @ E_RL @ Kinv
    F_RL /= F_RL[-1, -1]
    return F_RL


@pytest.fixture
def F_LR(F_RL):
    return F_RL.T


@pytest.fixture
def num_random_points():
    return 10


@pytest.fixture
def random_points3d_L(num_random_points):
    g = np.random.default_rng(42)
    points = g.random((num_random_points, 3))
    points[:, :2] *= 2.0
    points[:, 2] += 10.0
    return points


@pytest.fixture
def image_points_L(random_points3d_L: np.ndarray, camera_matrix: CameraMatrix):
    return camera_matrix.project(random_points3d_L)


@pytest.fixture
def image_points_R(
    random_points3d_L: np.ndarray, camera_matrix: CameraMatrix, R_RL: Rotation, t_RL: np.ndarray
):
    return camera_matrix.project(random_points3d_L @ R_RL.as_matrix().T + t_RL)


def test_fundamental_fixture(image_points_L, image_points_R, F_RL):
    assert np.allclose(
        np.diag(homogenize(image_points_L) @ F_RL @ homogenize(image_points_R).T),
        0.0,
        atol=1e-6,
    )


def test_epilines_convergence(F_RL, image_points_L, image_points_R):
    epilines_L = get_epilines(image_points_R, F_RL)
    estimated_epipole_L = intersect_lines_2d(epilines_L)
    epipole_L = get_epipole(F_RL)
    assert np.linalg.norm(estimated_epipole_L - epipole_L) < 0.1

    epilines_R = get_epilines(image_points_L, F_RL.T)
    estimated_epipole_R = intersect_lines_2d(epilines_R)
    epipole_R = get_epipole(F_RL.T)
    assert np.linalg.norm(estimated_epipole_R - epipole_R) < 0.1


def test_epipole(F_RL):
    assert np.allclose(get_homo_epipole(F_RL) @ F_RL, 0.0, atol=1e-7)
    assert np.allclose(F_RL @ get_homo_epipole(F_RL.T), 0.0, atol=1e-7)


def test_fundamental_estimation(
    image_points_L: np.ndarray,
    image_points_R: np.ndarray,
    F_RL: np.ndarray,
):
    fundamental_solver = Fundamental(FundamentalOptions(), image_points_L, image_points_R)
    estimated_F_RL = fundamental_solver.get_F_RL()
    assert np.linalg.norm(F_RL - estimated_F_RL) < 1e-4
    assert np.allclose(np.linalg.det(estimated_F_RL), 0.0, 1e-15)


def test_essential_estimation(
    image_points_L: np.ndarray,
    image_points_R: np.ndarray,
    camera_matrix: CameraMatrix,
    E_RL: np.ndarray,
    R_RL: Rotation,
    t_RL: np.ndarray,
):
    essential_solver = Essential(
        FundamentalOptions(), image_points_L, image_points_R, camera_matrix
    )
    R_RL = R_RL.as_matrix()
    estimated_E_RL = essential_solver.get_E_RL()
    estimated_R_RL, estimated_t_RL = essential_solver.get_pose_RL()

    # Pick a non-zero entry for normalization, by construction, the last entry is 0 in the matrix.
    normalized_E_RL = E_RL.copy()
    normalized_E_RL /= normalized_E_RL[0, 1]
    normalized_estimated_E_RL = estimated_E_RL.copy()
    normalized_estimated_E_RL /= normalized_estimated_E_RL[0, 1]

    assert np.linalg.norm(normalized_E_RL - normalized_estimated_E_RL) < 1e-3
    assert np.linalg.norm(R_RL - estimated_R_RL) < 1e-4
    assert np.linalg.norm(t_RL - estimated_t_RL) < 0.0002


def test_epipole_convergence(F_RL, image_points_L, image_points_R):
    """Test the epipole convergence w.r.t. the given dummy data.

    The error is coming from the fact that the real epipole is not all real.
    """
    lines_L = get_epilines(image_points_R, F_RL)
    lines_R = get_epilines(image_points_L, F_RL.T)
    intersection_L = intersect_lines_2d(lines_L)
    intersection_R = intersect_lines_2d(lines_R)
    epipole_L = get_epipole(F_RL)
    epipole_R = get_epipole(F_RL.T)
    np.testing.assert_allclose(intersection_L, epipole_L, atol=1e-2)
    np.testing.assert_allclose(intersection_R, epipole_R, atol=1e-2)


def test_epipole_translation(F_RL, image_points_L, image_points_R):
    for image_point_L, image_point_R in zip(image_points_L, image_points_R):
        T_L = np.eye(3)
        T_L[:2, 2] = -image_point_L
        Tinv_L = np.linalg.inv(T_L)

        T_R = np.eye(3)
        T_R[:2, 2] = -image_point_R
        Tinv_R = np.linalg.inv(T_R)

        translated_F_RL = Tinv_L.T @ F_RL @ Tinv_R
        translated_homo_epipole_L = get_homo_epipole(translated_F_RL)
        translated_homo_epipole_R = get_homo_epipole(translated_F_RL.T)
        np.testing.assert_allclose(translated_F_RL @ translated_homo_epipole_R, 0.0, atol=1e-15)
        np.testing.assert_allclose(translated_homo_epipole_L @ translated_F_RL, 0.0, atol=1e-15)


def test_epipole_rotation(F_RL):
    # Rotate left and right epipole
    homo_epipole_L = get_homo_epipole(F_RL)
    homo_epipole_L /= np.linalg.norm(homo_epipole_L[:2])
    R_L = np.eye(3)
    R_L[:2, :2] = [
        [homo_epipole_L[0], homo_epipole_L[1]],
        [-homo_epipole_L[1], homo_epipole_L[0]],
    ]
    assert np.allclose(np.linalg.det(R_L), 1.0)
    homo_epipole_R = get_homo_epipole(F_RL.T)
    homo_epipole_R /= np.linalg.norm(homo_epipole_R[:2])
    R_R = np.eye(3)
    R_R[:2, :2] = [
        [homo_epipole_R[0], homo_epipole_R[1]],
        [-homo_epipole_R[1], homo_epipole_R[0]],
    ]
    assert np.allclose(np.linalg.det(R_R), 1.0)

    rotated_F_RL = R_L @ F_RL @ R_R.T

    assert np.allclose([1.0, 0.0, homo_epipole_L[-1]] @ rotated_F_RL, 0.0, 1e-15)
    assert np.allclose(rotated_F_RL @ [1.0, 0.0, homo_epipole_R[-1]], 0.0, 1e-15)


def _ensure_rigid_body_motion(F_RL, x_L, x_R):
    """Test the rigid motion logic for transforming the epipole to (1, 0, f) and (1, 0, f').

    This is a Python version of the same logic implemented in C++.
    """
    # Sanity check
    np.testing.assert_allclose(np.r_[x_L, 1.0] @ F_RL @ np.r_[x_R, 1.0], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        get_homo_epipole(F_RL) @ F_RL @ get_homo_epipole(F_RL.T), 0.0, atol=1e-15
    )

    # Move the image point to (0, 0, 1).
    T_L = np.eye(3)
    T_L[:2, 2] = -x_L
    Tinv_L = np.linalg.inv(T_L)

    T_R = np.eye(3)
    T_R[:2, 2] = -x_R
    Tinv_R = np.linalg.inv(T_R)

    # Construct rotation matrix to move the epipoles to (1, 0, f).
    translated_F_RL = Tinv_L.T @ F_RL @ Tinv_R
    translated_homo_epipole_L = get_homo_epipole(translated_F_RL)
    norm_L = np.linalg.norm(translated_homo_epipole_L[:2])
    translated_homo_epipole_L /= norm_L
    np.testing.assert_allclose(translated_F_RL.T @ translated_homo_epipole_L, 0.0, atol=1e-15)

    translated_homo_epipole_R = get_homo_epipole(translated_F_RL.T)
    norm_R = np.linalg.norm(translated_homo_epipole_R[:2])
    translated_homo_epipole_R /= norm_R
    np.testing.assert_allclose(translated_F_RL @ translated_homo_epipole_R, 0.0, atol=1e-15)

    R_L = np.eye(3)
    R_L[:2, :2] = [
        [translated_homo_epipole_L[0], translated_homo_epipole_L[1]],
        [-translated_homo_epipole_L[1], translated_homo_epipole_L[0]],
    ]
    np.testing.assert_allclose(np.linalg.det(R_L), 1.0, 1e-15)
    transformed_homo_epipole_L = R_L @ translated_homo_epipole_L
    expected_homo_epipole_L = np.asarray([1.0, 0.0, translated_homo_epipole_L[2]])
    norm = np.linalg.norm(transformed_homo_epipole_L - expected_homo_epipole_L)
    assert np.allclose(norm, 0.0, atol=1e-15)

    R_R = np.eye(3)
    R_R[:2, :2] = [
        [translated_homo_epipole_R[0], translated_homo_epipole_R[1]],
        [-translated_homo_epipole_R[1], translated_homo_epipole_R[0]],
    ]
    transformed_homo_epipole_R = R_R @ translated_homo_epipole_R
    expected_homo_epipole_R = np.asarray([1.0, 0.0, translated_homo_epipole_R[2]])
    norm = np.linalg.norm(transformed_homo_epipole_R - expected_homo_epipole_R)
    np.testing.assert_allclose(np.linalg.det(R_R), 1.0, 1e-15)

    transformed_F_RL = R_L @ translated_F_RL @ R_R.T
    RTFTR_RL = R_L @ Tinv_L.T @ F_RL @ Tinv_R @ R_R.T
    np.testing.assert_allclose(transformed_F_RL - RTFTR_RL, 0.0, atol=1e-15)

    np.testing.assert_allclose(
        norm_L * translated_homo_epipole_L @ translated_F_RL @ translated_homo_epipole_R * norm_R,
        0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        transformed_homo_epipole_L @ R_L @ translated_F_RL @ R_R.T @ transformed_homo_epipole_R,
        0.0,
        atol=1e-15,
    )

    assert np.allclose(transformed_F_RL @ transformed_homo_epipole_R, 0.0, atol=1e-15)
    assert np.allclose(transformed_F_RL.T @ expected_homo_epipole_L, 0.0, atol=1e-15)

    a = transformed_F_RL[1, 1]
    b = transformed_F_RL[2, 1]
    c = transformed_F_RL[1, 2]
    d = transformed_F_RL[2, 2]
    f1 = expected_homo_epipole_L[2]
    f2 = expected_homo_epipole_R[2]

    expected_F_RL = np.asarray(
        [
            [f1 * f2 * d, -f1 * b, -f1 * d],
            [-f2 * c, a, c],
            [-f2 * d, b, d],
        ]
    )
    assert np.allclose(expected_F_RL, transformed_F_RL, atol=1e-15)


def test_epipole_rigid_motion(F_RL, image_points_L, image_points_R):
    """Test the rigid motion logic for transforming the epipole to (1, 0, f) and (1, 0, f')."""
    for image_point_L, image_point_R in zip(image_points_L, image_points_R):
        _ensure_rigid_body_motion(F_RL, image_point_L, image_point_R)


def test_estimation_manual_data_association(leuven_stereo_data_pack):
    """Test the convergence of intersections of reprojected epilines on the other view
    using manually associated points."""

    points_L = leuven_stereo_data_pack.manual_points_L
    points_R = leuven_stereo_data_pack.manual_points_R
    options = FundamentalOptions()
    options.atol = 2.8  # NOTE(kun): error for manually annotated points.
    fundamental = Fundamental(options, points_L, points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = fundamental.get_inlier_indices()
    lines_L = get_epilines(points_R[inlier_indices], F_RL)
    lines_R = get_epilines(points_L[inlier_indices], F_RL.T)

    np.testing.assert_allclose(get_epipole(F_RL), intersect_lines_2d(lines_L), atol=0.2)
    np.testing.assert_allclose(get_epipole(F_RL.T), intersect_lines_2d(lines_R), atol=0.2)


def test_estimation_with_detected_keypoints(leuven_stereo_data_pack):
    """Test the convergence of intersections of reprojected epilines on the other view
    using detected SIFT keypoints."""

    points_L = leuven_stereo_data_pack.points_L
    points_R = leuven_stereo_data_pack.points_R
    fundamental = Fundamental(FundamentalOptions(), points_L, points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = fundamental.get_inlier_indices()
    lines_L = get_epilines(points_R[inlier_indices], F_RL)
    lines_R = get_epilines(points_L[inlier_indices], F_RL.T)

    np.testing.assert_allclose(get_epipole(F_RL), intersect_lines_2d(lines_L), atol=0.2)
    np.testing.assert_allclose(get_epipole(F_RL.T), intersect_lines_2d(lines_R), atol=0.2)


def test_two_view_reprojection_error(leuven_stereo_data_pack, stereo_reprojection_rms_threshold):
    camera_matrix = leuven_stereo_data_pack.camera_matrix
    points_L = leuven_stereo_data_pack.manual_points_L
    points_R = leuven_stereo_data_pack.manual_points_R

    essential = Essential(FundamentalOptions(atol=5.0), points_L, points_R, camera_matrix)

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

    lines_L = get_epilines(x_R=points_inliers_R, F_RL=F_RL)
    warped_lines_L = lines_L @ np.linalg.inv(H_L)
    np.testing.assert_allclose((warped_lines_L[:, 1] / warped_lines_L[:, 0]).ptp(), 0.0)

    lines_R = get_epilines(x_L=points_inliers_L, F_RL=F_RL.T)
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
