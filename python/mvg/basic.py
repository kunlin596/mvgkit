#!/usr/bin/env python3 -B
"""This module includes basic math functions"""
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


class UnitConverter:

    METER_TO_MILLIMETER = 1000.0
    MILLIMETER_TO_METER = 0.001

    @staticmethod
    def T_to_meter(T: np.ndarray):
        assert T.shape == (4, 4)
        T = T.copy()
        T[:3, 3] *= UnitConverter.MILLIMETER_TO_METER
        return T


def homogeneous_transformation(R: np.ndarray, t: np.ndarray):
    assert R.shape == (3, 3)
    assert len(t) == 3
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


class SE3:
    _R: Rotation
    _t: np.ndarray

    def __init__(self, R: Rotation, t: np.ndarray):
        self._R = R
        self._t = t.reshape(-1)

    @property
    def R(self):
        return self._R

    @property
    def t(self):
        return self._t

    def as_homogeneous_matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.R.as_matrix()
        T[:3, 3] = self.t
        return T

    def as_augmented_matrix(self):
        return np.hstack([self.R.as_matrix(), self.t.reshape(-1, 1)])

    def as_pose(self):
        return np.r_[self.R.as_quat(), self.t]

    def __repr__(self) -> str:
        return f"SE3(R={self.R.as_quat()}, t={self.t})"


def get_symbolic_rodrigues_rotmat(*, r1: sp.Symbol, r2: sp.Symbol, r3: sp.Symbol):
    # NOTE: Rodrigues's rotation formula, check more
    rvec = np.asarray([r1, r2, r3])
    theta = sp.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2)
    rvec /= theta
    K = sp.Matrix(np.cross(rvec, np.eye(3))).T
    return sp.eye(3) + sp.sin(theta) * K + (1.0 - sp.cos(theta)) * K @ K
