"""This module implements a sketch of the EPnP algorithm.
"""
import itertools
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from mvgkit.common.camera import CameraMatrix
from pymvgkit_common import get_barycentric_coords_3d, get_rigid_body_motion


def _compute_control_points(points_W):
    centroid = np.mean(points_W, axis=0)
    shifted_points_W = points_W - centroid
    cov = shifted_points_W.T @ shifted_points_W
    # In ascending order
    _, eigenvectors = np.linalg.eig(cov)
    # KUN: to match the eigenvalues in C++ for now.
    eigenvectors[:, 0] = -eigenvectors[:, 0]
    eigenvectors[:, 1] = -eigenvectors[:, 1]
    control_points_W = np.tile(centroid, (4, 1))
    control_points_W[1:, :] += eigenvectors.T
    return control_points_W


def _compose_M(
    points_W: np.ndarray,
    coords: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: CameraMatrix,
):
    M = np.empty((points_W.shape[0] * 2, 12))
    fx, fy, cx, cy, _ = camera_matrix.as_array().tolist()
    for i in range(0, len(M), 2):
        coord = coords[i // 2]
        image_point = image_points[i // 2]
        M[i] = list(
            itertools.chain(
                *[[coord[j] * fx, 0.0, coord[j] * (cx - image_point[0])] for j in range(0, 4)]
            )
        )
        M[i + 1] = list(
            itertools.chain(
                *[[0.0, coord[j] * fy, coord[j] * (cy - image_point[1])] for j in range(0, 4)]
            )
        )
    return M


def _get_beta_1(L, rho):
    """
    From [b00, b01, b02, b03, b11, b12, b13, b22, b23, b33],
    pick [b00, b01, b02, b03,                             ].
    """
    A = L[:, [0, 1, 2, 3]]
    x = np.linalg.inv(A.T @ A) @ A.T @ rho
    betas = np.empty(4)
    if x[0] < 0.0:
        x = -x
    betas[0] = math.sqrt(x[0])
    betas[1] = x[1] / betas[0]
    betas[2] = x[2] / betas[0]
    betas[3] = x[3] / betas[0]
    print("x", x)
    print("betas ", betas)
    return betas


def _get_beta_2(L, rho):
    """
    From [b00, b01, b02, b03, b11, b12, b13, b22, b23, b33],
    pick [b00, b01,           b11.                        ].
    """
    A = L[:, [0, 1, 4]]
    x = np.linalg.inv(A.T @ A) @ A.T @ rho
    betas = np.empty(4)
    betas[2] = betas[3] = 0.0
    if x[0] < 0.0:
        betas[0] = math.sqrt(-x[0])
        betas[1] = math.sqrt(-x[2]) if x[2] < 0.0 else 0.0
    else:
        betas[0] = math.sqrt(x[0])
        betas[1] = math.sqrt(x[2]) if x[2] > 0.0 else 0.0
    if x[1] < 0.0:
        betas[0] = -betas[0]
    print("x", x)
    print("betas ", betas)
    return betas


def _get_beta_3(L, rho):
    """
    From [b00, b01, b02, b03, b11, b12, b13, b22, b23, b33],
    pick [b00, b01, b02,      b11, b12,                   ].
    """
    A = L[:, [0, 1, 2, 4, 5]]
    x = np.linalg.inv(A.T @ A) @ A.T @ rho
    betas = np.empty(4)

    if x[0] < 0.0:
        betas[0] = math.sqrt(-x[0])
        betas[1] = math.sqrt(-x[3]) if x[3] < 0.0 else 0.0
    else:
        betas[0] = math.sqrt(x[0])
        betas[1] = math.sqrt(x[3]) if x[3] > 0.0 else 0.0
    if x[1] < 0.0:
        betas[0] = -betas[0]
    betas[2] = x[2] / betas[0]
    betas[3] = 0.0
    print("x", x)
    print("betas ", betas)
    return betas


def _get_reprojection_error(points_C, camera_matrix, image_points):
    return np.linalg.norm(camera_matrix.project(points_C) - image_points, axis=1).sum()


def _optimize_pose(initial_pose, points_W, image_points, camera_matrix):
    def _residual(x):
        R = Rotation.from_rotvec(x[:3])
        t = x[3:]
        points_C = R.apply(points_W) + t
        return (camera_matrix.project(points_C) - image_points).reshape(-1)

    result = least_squares(_residual, x0=initial_pose)
    if result["success"]:
        return result["x"]
    return initial_pose


def solve_epnp(points_W: np.ndarray, image_points: np.ndarray, camera_matrix: CameraMatrix):
    assert len(points_W) == len(image_points)
    control_points_W = _compute_control_points(points_W)

    coords = np.empty((points_W.shape[0], 4))
    for i, point_W in enumerate(points_W):
        coords[i] = get_barycentric_coords_3d(control_points_W, point_W)

    M = _compose_M(points_W, coords, image_points, camera_matrix)
    MtM = M.transpose() @ M
    eigenvalues, eigenvectors = np.linalg.eig(MtM)

    # In column major, convert it to row major to be consistent
    eigenvector_candidates = eigenvectors.T[eigenvalues.argsort()][:4]
    v1 = eigenvector_candidates[0]
    v2 = eigenvector_candidates[1]
    v3 = eigenvector_candidates[2]
    v4 = eigenvector_candidates[3]

    # Build L and rho
    i = [0, 0, 0, 1, 1, 2]
    j = [1, 2, 3, 2, 3, 3]
    L = np.empty((6, 10))
    rho = np.empty(6)

    for k in range(6):
        i1 = i[k] * 3
        i2 = i[k] * 3 + 3

        j1 = j[k] * 3
        j2 = j[k] * 3 + 3

        s1 = v1[i1:i2] - v1[j1:j2]
        s2 = v2[i1:i2] - v2[j1:j2]
        s3 = v3[i1:i2] - v3[j1:j2]
        s4 = v4[i1:i2] - v4[j1:j2]

        L[k] = [
            s1 @ s1,
            2.0 * s1 @ s2,
            2.0 * s1 @ s3,
            2.0 * s1 @ s4,
            s2 @ s2,
            2.0 * s2 @ s3,
            2.0 * s2 @ s4,
            s3 @ s3,
            2.0 * s3 @ s4,
            s4 @ s4,
        ]

        rho[k] = np.linalg.norm(control_points_W[i[k]] - control_points_W[j[k]])

    beta_fns = [
        _get_beta_1,
        _get_beta_2,
        _get_beta_3,
    ]
    poses = np.empty((3, 6))
    errors = np.ones(3) * np.inf

    np.set_printoptions(suppress=True, precision=12, linewidth=150)

    for i, beta_fn in enumerate(beta_fns):
        beta = beta_fn(L, rho)
        points_C = (beta.T @ eigenvector_candidates).reshape(4, 3)
        initialPose = get_rigid_body_motion(control_points_W, points_C)
        poses[i] = _optimize_pose(initialPose, points_W, image_points, camera_matrix)
        R = Rotation.from_rotvec(poses[i][:3])
        t = poses[i][3:]
        errors[i] = _get_reprojection_error(R.apply(points_W) + t, camera_matrix, image_points)

    best_pose = poses[errors.argmin()]

    return Rotation.from_rotvec(best_pose[:3]).as_matrix(), best_pose[3:]
