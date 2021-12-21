#!/usr/bin/env python3 -B
"""This module includes basic math functions"""
from dataclasses import dataclass
from math import sqrt
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation


def homogeneous(p: np.ndarray, axis: int = 1, omega: float = 1.0):
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


def get_normalization_matrix_2d(points: np.ndarray) -> np.ndarray:
    """
    Normalize input data is recommended by the paper below,
    R. I. Hartley. In defense of the eight-point algorithm. IEEE Trans. Pattern
    Analysis and Machine Intelligence, 19(6):580–593, 1997.

    NOTE: that there is a `sqrt(2)` appeared in the nominator but the reason is not clean to me.
    Perhaps it's just to scale the point down a bit.

    NOTE: More sophisticated method such as PCA could be used to normalization as well.
    """
    # Build normalization matrix
    N = np.eye(3)
    sqrt_2 = sqrt(2)
    std_x, std_y = points.std(axis=0)
    std_x = sqrt_2 / std_x
    std_y = sqrt_2 / std_y
    mean_x, mean_y = points.mean(axis=0)
    N[0, 0] = std_x
    N[1, 1] = std_y
    N[:2, 2] = [-std_x * mean_x, -std_y * mean_y]
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
