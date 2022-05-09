#!/usr/bin/env python3
"""This module includes basic math functions"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List

import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation


def get_points_mesh(*, width, height, step_size=1.0) -> np.ndarray:
    return np.transpose(np.mgrid[:width, :height], (2, 1, 0)).reshape(-1, 2) * step_size


def is_single_point_3d(points: np.ndarray):
    return len(points.reshape(-1)) == 3


def is_point_array_3d(points: np.ndarray):
    return len(points.shape) == 2 and points.shape[-1] == 3


def homogenize(p: np.ndarray | list, axis: int = 1, omega: float = 1.0) -> np.ndarray | list:
    is_not_ndarray = not isinstance(p, np.ndarray)

    if is_not_ndarray:
        p = np.asarray(p)

    result = None
    if len(p.shape) == 1:
        result = np.r_[p, 1.0]

    elif len(p.shape) == 2:
        if axis == 0:
            result = np.vstack([p, np.ones(shape=(1, len(p)), dtype=p.dtype) * omega])

        elif axis == 1:
            result = np.hstack([p, np.ones(shape=(len(p), 1), dtype=p.dtype) * omega])

    if is_not_ndarray:
        return result.tolist()
    return result


def dehomogenize(p: np.ndarray | list, axis: int = 1) -> np.ndarray | list:
    is_not_ndarray = not isinstance(p, np.ndarray)
    if is_not_ndarray:
        p = np.asarray(p)

    result = None
    if len(p.shape) == 1:
        result = (p / p[-1])[:-1]
    elif len(p.shape) == 2:
        if axis == 0:
            result = (p / p[-1, :].reshape(1, -1))[:-1, :]
        elif axis == 1:
            result = (p / p[:, -1].reshape(-1, 1))[:, :-1]

    if is_not_ndarray:
        return result.tolist()
    return result


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


def transform_points(T: np.ndarray, points):
    assert T.shape == (4, 4)
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def homogeneous_transformation(R: np.ndarray, t: np.ndarray):
    assert R.shape == (3, 3)
    assert len(t) == 3
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


@dataclass
class SE3:
    R: Rotation | None = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    t: np.ndarray | None = np.array([0.0, 0.0, 0.0])

    @staticmethod
    def from_quat_pose(pose: np.ndarray):
        assert len(pose) == 7
        return SE3(Rotation.from_quat(pose[:4]), pose[4:])

    @staticmethod
    def from_rotvec_pose(pose: np.ndarray):
        assert len(pose) == 6
        return SE3(Rotation.from_rotvec(pose[:3]), pose[3:])

    @staticmethod
    def from_rotmat_tvec(rotmat, tvec):
        return SE3(Rotation.from_matrix(rotmat), tvec)

    def as_matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.R.as_matrix()
        T[:3, 3] = self.t
        return T

    @staticmethod
    def identity():
        return SE3.from_rotvec_pose(np.zeros(6))

    def as_augmented_matrix(self):
        return np.hstack([self.R.as_matrix(), self.t.reshape(-1, 1)])

    def as_quat_pose(self):
        return np.r_[self.R.as_quat().reshape(-1), self.t.reshape(-1)]

    def as_rotvec_pose(self):
        return np.r_[self.R.as_rotvec().reshape(-1), self.t.reshape(-1)]

    def inv(self):
        R_inv = self.R.inv()
        return SE3(R_inv, -R_inv.as_matrix() @ self.t)

    def __repr__(self) -> str:
        return f"SE3(R={self.R.as_quat()}, t={self.t})"

    def __matmul__(self, right_term):  # noqa: C901
        if isinstance(right_term, List):
            right_term = np.asarray(right_term)

        if isinstance(right_term, np.ndarray):
            if len(right_term.shape) == 1:
                if len(right_term) == 3:
                    return self.R.as_matrix() @ right_term + self.t
                if len(right_term) == 4:
                    return self.as_matrix() @ right_term
            else:
                if right_term.shape[0] == 3:
                    return self.R.as_matrix() @ right_term + self.t.reshape(-1, 1)
                elif right_term.shape[0] == 4:
                    return self.as_matrix() @ right_term
                elif right_term.shape[-1] == 3:
                    return right_term @ self.R.as_matrix().T + self.t
                elif right_term.shape[-1] == 4:
                    return right_term @ self.as_matrix().T
        elif isinstance(right_term, SE3):
            result = self.as_matrix() @ right_term.as_matrix()
            return SE3.from_rotmat_tvec(result[:3, :3], result[:3, 3])

    @property
    def T(self):
        return self.as_matrix().T

    def transpose(self):
        return self.T


@dataclass
class SkewSymmetricMatrix3d:

    vec: np.ndarray = np.zeros(3)

    def __post_init__(self):
        mat = self.as_matrix()
        assert np.allclose(
            mat, -mat.T
        ), "This is not a valid skew symmetric matrix, perhaps it's initialized directly from an invalid matrix!"

    @staticmethod
    def from_vec(vec):
        vec = np.asarray(vec)
        assert len(vec) == 3
        return SkewSymmetricMatrix3d(np.asarray(vec))

    @staticmethod
    def from_matrix(mat):
        mat = np.asarray(mat)
        assert mat.shape == (3, 3)
        return SkewSymmetricMatrix3d([mat[2, 1], mat[0, 2], mat[1, 0]])

    def as_vec(self):
        return np.asarray(self.vec)

    def as_matrix(self):
        self.vec = np.asarray(self.vec)
        return np.cross(self.vec.reshape(1, -1), np.eye(3, dtype=self.vec.dtype)).T

    def T(self):
        return SkewSymmetricMatrix3d(-self.vec)


@dataclass
class PluckerMatrix:

    p1: np.ndarray = np.zeros(4)
    p2: np.ndarray = np.zeros(4)

    def __post_init__(self):
        assert len(self.p1) == 4
        assert len(self.p2) == 4
        self.p1 = np.asarray(self.p1)
        self.p2 = np.asarray(self.p2)
        mat = self.as_matrix()
        assert np.allclose(mat, -mat.T), "This is not a valid Plucker matrix!"

    @staticmethod
    def from_vecs(p1, p2):
        return SkewSymmetricMatrix3d(p1, p2)

    def as_vecs(self):
        return self.p1, self.p2

    def as_matrix(self):
        A = self.p1.reshape(-1, 4)
        B = self.p1.reshape(-1, 4)
        return A @ B.T - B @ A.T


def get_symbolic_rodrigues_rotmat(*, r1: sp.Symbol, r2: sp.Symbol, r3: sp.Symbol):
    # NOTE: Rodrigues's rotation formula, check more
    rvec = np.asarray([r1, r2, r3])
    theta = sp.sqrt(r1**2 + r2**2 + r3**2)
    rvec /= theta
    K = sp.Matrix(np.cross(rvec, np.eye(3))).T
    return sp.eye(3) + sp.sin(theta) * K + (1.0 - sp.cos(theta)) * K @ K


def line_distance_2d(*, points_2d: np.ndarray, line: np.ndarray):
    """Compute the distances of points to line"""
    return homogenize(points_2d) @ line / np.linalg.norm(line[:2])


def normalize_vectors(x, axis=None):
    x = np.asarray(x)
    if len(x.shape) == 1:
        return x / np.linalg.norm(x)
    else:
        if axis is None:
            axis = 1
        return x / np.linalg.norm(x, axis=axis)


def get_line_points_in_image(line: np.ndarray, width: float, height: float):
    """Compute line points using line in the form of [a, b, c] in the image of shape [height, width]."""
    assert len(line) == 3
    x = np.arange(0, width, 0.1)
    y = (-line[2] - x * line[0]) / line[1]
    isvalid = (0 <= y) & (y < height)
    return np.vstack([x[isvalid], y[isvalid]]).T
