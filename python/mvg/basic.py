#!/usr/bin/env python3 -B
"""This module includes basic math functions"""
from dataclasses import dataclass
from math import sqrt
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation


def homogeneous(p: np.ndarray, axis: int = 1, omega: float = 1.0) -> np.ndarray:
    if len(p.shape) == 1:
        return np.r_[p, 1.0]
    elif len(p.shape) == 2:
        if axis == 0:
            return np.vstack([p, np.ones(shape=(1, len(p)), dtype=p.dtype) * omega])
        elif axis == 1:
            return np.hstack([p, np.ones(shape=(len(p), 1), dtype=p.dtype) * omega])


def transform_points(T: np.ndarray, points):
    assert T.shape == (4, 4)
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def skew_symmetric_matrix(vec):
    return np.cross(vec.reshape(1, -1), np.eye(3, dtype=vec.dtype))


def get_isotropic_scaling_matrix_2d(points: np.ndarray, target_distance=None) -> np.ndarray:
    """
    Scale the points such that they are zero-meaned and the mean distance from origin is `target_distance`.

    See 5.1 of the paper below for isotropic scaling,

    R. I. Hartley. In defense of the eight-point algorithm. IEEE Trans. Pattern
    Analysis and Machine Intelligence, 19(6):580–593, 1997.
    """

    if target_distance is None:
        target_distance = sqrt(2)

    N = np.eye(3)
    mean = points.mean(axis=0)
    sum_distances = np.sum(np.linalg.norm(points - mean, axis=1))
    scale = target_distance * len(points) / sum_distances

    N[0, 0] = scale
    N[1, 1] = scale
    N[:2, 2] = [-mean[0] * scale, -mean[1] * scale]
    return N


def get_nonisotropic_scaling_matrix_2d(points: np.ndarray) -> np.ndarray:
    """
    See 5.2 of the paper below for non-isotropic scaling,

    R. I. Hartley. In defense of the eight-point algorithm. IEEE Trans. Pattern
    Analysis and Machine Intelligence, 19(6):580–593, 1997.
    """
    N = np.eye(3)
    sqrt_2 = sqrt(2)
    std_x, std_y = points.std(axis=0)
    mean_x, mean_y = points.mean(axis=0)

    x_scale = sqrt_2 / std_x
    y_scale = sqrt_2 / std_y
    N[0, 0] = x_scale
    N[1, 1] = y_scale
    N[:2, 2] = [-mean_x * x_scale, -mean_y * y_scale]
    return N


def homogeneous_transformation(R: np.ndarray, t: np.ndarray):
    assert R.shape == (3, 3)
    assert len(t) == 3
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


@dataclass
class SE3:
    R: Rotation = Rotation.from_quat([0, 0, 0, 1])
    t: np.ndarray = np.zeros(3)

    @staticmethod
    def from_quat_pose(pose: np.ndarray):
        assert len(pose) == 7
        return SE3(Rotation.from_quat(pose[:4]), pose[4:])

    @staticmethod
    def from_rotvec_pose(pose: np.ndarray):
        assert len(pose) == 6
        return SE3(Rotation.from_rotvec(pose[:3]), pose[3:])

    def as_homogeneous_matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.R.as_matrix()
        T[:3, 3] = self.t
        return T

    def as_augmented_matrix(self):
        return np.hstack([self.R.as_matrix(), self.t.reshape(-1, 1)])

    def as_quat_pose(self):
        return np.r_[self.R.as_quat().reshape(-1), self.t.reshape(-1)]

    def as_rotvec_pose(self):
        return np.r_[self.R.as_rotvec().reshape(-1), self.t.reshape(-1)]

    def __repr__(self) -> str:
        return f"SE3(R={self.R.as_quat()}, t={self.t})"


def get_symbolic_rodrigues_rotmat(*, r1: sp.Symbol, r2: sp.Symbol, r3: sp.Symbol):
    # NOTE: Rodrigues's rotation formula, check more
    rvec = np.asarray([r1, r2, r3])
    theta = sp.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    rvec /= theta
    K = sp.Matrix(np.cross(rvec, np.eye(3))).T
    return sp.eye(3) + sp.sin(theta) * K + (1.0 - sp.cos(theta)) * K @ K


def line_distance_2d(*, points_2d: np.ndarray, line: np.ndarray):
    """Compute the distances of points to line"""
    return homogeneous(points_2d) @ line / np.linalg.norm(line[:2])
