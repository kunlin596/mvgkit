#!/usr/bin/env python3
from math import pi
import unittest

import numpy as np
from mvg.basic import SE3, PluckerMatrix, SkewSymmetricMatrix3d, dehomogenize, homogenize
from numpy import allclose

np.set_printoptions(suppress=True, precision=15, linewidth=120)


class BasicTests(unittest.TestCase):
    @property
    def row_points(self):
        return np.random.random((10, 3))

    @property
    def col_points(self):
        return np.random.random((3, 10))

    @property
    def T(self):
        return SE3.from_rotvec_pose(pose=np.random.random(6))

    def test_homogenize(self):
        # Single point
        p = homogenize([1, 1])
        assert isinstance(p, list)
        assert np.allclose(p[-1], 1.0)

        p = homogenize(np.array([1, 1]))
        assert isinstance(p, np.ndarray)
        assert np.allclose(p[-1], 1.0)

        # Row wise homogeneous
        p = homogenize([[1, 1], [1, 1]], axis=0, omega=2.0)
        assert isinstance(p, list)
        p = np.asarray(p)
        assert p.shape == (3, 2)
        assert np.allclose(np.asarray(p[-1, :]), 2.0)

        p = homogenize([[1, 1], [1, 1]], axis=1, omega=2.0)
        assert isinstance(p, list)
        p = np.asarray(p)
        assert p.shape == (2, 3)
        assert np.allclose(np.asarray(p[:, -1]), 2.0)

        # Column wise homogeneous
        p = homogenize(np.array([[1, 1], [1, 1]]), axis=0, omega=2.0)
        assert isinstance(p, np.ndarray)
        assert p.shape == (3, 2)
        assert np.allclose(np.asarray(p[-1, :]), 2.0)

        p = homogenize(np.array([[1, 1], [1, 1]]), axis=1, omega=2.0)
        assert isinstance(p, np.ndarray)
        assert p.shape == (2, 3)
        assert np.allclose(np.asarray(p[:, -1]), 2.0)

    def test_dehomogenize(self):
        # Single point
        p = dehomogenize([2, 2, 2])
        assert isinstance(p, list)
        assert len(p) == 2
        assert np.allclose(p, [1, 1])

        # Row dehomogenization
        p = [[2, 2, 2], [4, 8, 2], [8, 4, 4]]
        p2 = dehomogenize(p, axis=0)

        assert isinstance(p, list)
        p = np.asarray(p)
        p2 = np.asarray(p2)
        assert p2.shape == (2, 3)
        assert np.allclose(p2, p[:2, :] / p[-1, :])

        p = np.array([[2, 2, 2], [4, 8, 2], [8, 4, 4]])
        p2 = dehomogenize(p, axis=0)
        assert isinstance(p, np.ndarray)
        assert p2.shape == (2, 3)
        assert np.allclose(p2, p[:2, :] / p[-1, :])

        # Column dehomogenization
        p = [[2, 2, 2], [4, 8, 2], [8, 4, 4]]
        p2 = dehomogenize(p, axis=1)

        assert isinstance(p, list)
        p = np.asarray(p)
        p2 = np.asarray(p2)
        assert p2.shape == (3, 2)
        assert np.allclose(p2, p[:, :2] / p[:, -1].reshape(-1, 1))

        p = np.array([[2, 2, 2], [4, 8, 2], [8, 4, 4]])
        p2 = dehomogenize(p, axis=1)
        assert isinstance(p, np.ndarray)
        assert p2.shape == (3, 2)
        assert np.allclose(p2, p[:, :2] / p[:, -1].reshape(-1, 1))

    def test_SE3_simple_inv(self):
        T = SE3.from_rotvec_pose(pose=np.r_[0.0, 0.0, pi / 2, 0.0, 0.0, 0.0])
        assert np.allclose((T.inv() @ T).R.as_euler("zyx")[0], 0.0)

    def test_SE3_single_point_mul(self):
        T = self.T
        p = T.inv() @ (T @ [1.0, 1.0, 1.0])
        assert np.allclose(p, 1.0)

        p = T.inv() @ (T @ np.array([1.0, 1.0, 1.0]))
        assert np.allclose(p, 1.0)

    def test_SE3_row_points_mul(self):
        T = self.T
        row_points = self.row_points
        points2 = T.inv() @ (T @ row_points)
        assert points2.shape == row_points.shape
        assert np.allclose(points2, row_points)

        row_points = homogenize(row_points)
        points2 = T.inv() @ (T @ row_points)
        assert points2.shape == row_points.shape
        assert np.allclose(points2, row_points)

    def test_SE3_col_points_mul(self):
        T = self.T
        col_points = self.col_points
        points2 = T.inv() @ (T @ col_points)
        assert points2.shape == col_points.shape
        assert np.allclose(points2, col_points)

        col_points = homogenize(col_points)
        points2 = T.inv() @ (T @ col_points)
        assert points2.shape == col_points.shape
        assert np.allclose(points2, col_points)

    def test_skew_symmetric_matrix_3d(self):
        T1 = SkewSymmetricMatrix3d([1, 1, 1]).as_matrix()
        assert (T1.T == -T1).all()
        assert T1[0, 1] == -1
        assert T1[0, 2] == 1
        assert T1[1, 2] == -1

        T2 = SkewSymmetricMatrix3d.from_matrix(T1).as_vec()
        assert (T2 == [1, 1, 1]).all()

        assert allclose(
            SkewSymmetricMatrix3d.from_vec(T2).as_matrix(),
            SkewSymmetricMatrix3d.from_vec(-T2).as_matrix().T,
        )

    def test_PluckerMatrix(self):
        T = PluckerMatrix([1, 1, 1, 1], [2, 2, 2, 2]).as_matrix()
        assert np.allclose(T.T, -T)


if __name__ == "__main__":
    unittest.main()
