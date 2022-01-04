from math import pi
from mvg.basic import SE3, SkewSymmetricMatrix3d, PluckerMatrix, homogeneous
from numpy import allclose
import numpy as np
from pytest import fixture

np.set_printoptions(suppress=True, precision=15, linewidth=120)


@fixture()
def row_points():
    return np.random.random((10, 3))


@fixture()
def col_points():
    return np.random.random((3, 10))


@fixture
def T():
    return SE3.from_rotvec_pose(pose=np.random.random(6))


def test_SE3_simple_inv():
    T = SE3.from_rotvec_pose(pose=np.r_[0.0, 0.0, pi / 2, 0.0, 0.0, 0.0])
    assert np.allclose((T.inv() @ T).R.as_euler("zyx")[0], 0.0)


def test_SE3_single_point_mul(T):
    p = T.inv() @ (T @ [1.0, 1.0, 1.0])
    assert np.allclose(p, 1.0)

    p = T.inv() @ (T @ np.array([1.0, 1.0, 1.0]))
    assert np.allclose(p, 1.0)


def test_SE3_row_points_mul(T, row_points):
    points2 = T.inv() @ (T @ row_points)
    assert points2.shape == row_points.shape
    assert np.allclose(points2, row_points)

    row_points = homogeneous(row_points)
    points2 = T.inv() @ (T @ row_points)
    assert points2.shape == row_points.shape
    assert np.allclose(points2, row_points)


def test_SE3_col_points_mul(T, col_points):
    points2 = T.inv() @ (T @ col_points)
    assert points2.shape == col_points.shape
    assert np.allclose(points2, col_points)

    col_points = homogeneous(col_points)
    points2 = T.inv() @ (T @ col_points)
    assert points2.shape == col_points.shape
    assert np.allclose(points2, col_points)


def test_skew_symmetric_matrix_3d():
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


def test_PluckerMatrix():
    T = PluckerMatrix([1, 1, 1, 1], [2, 2, 2, 2]).as_matrix()
    assert np.allclose(T.T, -T)
