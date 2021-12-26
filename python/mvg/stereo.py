#!/usr/bin/env python3
"""This module implement stereo vision related algorithms."""


import cv2
from math import pi, sqrt
from mvg import basic

from scipy.optimize import least_squares
from scipy.spatial.transform.rotation import Rotation

import numpy as np

from mvg.basic import get_isotropic_scaling_matrix_2d, homogeneous


class Fundamental:
    """This class implements methods for computing fundamental matrix between stereo camera pairs.

    This implementation is expressing the right frame (R) in left frame (L).

    The constraints can be shown as below,

        F = x_L.T @ inv(K_L).T @ R_LR @ K_R @ x_R

    Here, x_L and x_R are the pixel coordinates in the frame (L) and (R) respectively,
    and K_L and L_R is the associated camera matrices. R_LR is the orientation of frame (R) expressed in frame (L).

    Also note that the rigid transformation is the pose of second camera expressed in left frame (L).

    For the sake of consistency, fundamental matrix is also adopting this notation
    denoting the direction of the camera movement.

    Depending on how you model your 3D points

    """

    @staticmethod
    def compute(*, x_L: np.ndarray, x_R: np.ndarray) -> np.ndarray:
        assert len(x_L) == len(x_R)

        N_L = get_isotropic_scaling_matrix_2d(x_L)
        N_R = get_isotropic_scaling_matrix_2d(x_R)

        x_L = homogeneous(x_L)
        x_R = homogeneous(x_R)

        normalized_x_L = x_L @ N_L.T
        normalized_x_R = x_R @ N_R.T

        F_LR = Fundamental._initialze(x_L=normalized_x_L, x_R=normalized_x_R)
        F_LR = Fundamental._optimize(x_L=normalized_x_L, x_R=normalized_x_R, initial_F_LR=F_LR)

        F_LR = N_L.T @ F_LR @ N_R
        F_LR = Fundamental._impose_F_rank(F_LR)
        F_LR /= F_LR[-1, -1]
        return F_LR

    @staticmethod
    def _linear_least_square_eight_point(*, x_L: np.ndarray, x_R: np.ndarray):
        """Solve homogeneous linear equations

        Since the objective function is A @ f = 0, w/o loss of generality,
        we can choose to set one of f to be 0, and solve the linear system.

        However, since we don't know which entry of f is non-zero,
        we need to test it 9 times.

        See 3.2.1 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """

        # Re-write x_L.T @ F @ x_R = 0 to A @ f = 0
        A = np.asarray([np.kron(p_L, p_R) for (p_L, p_R) in zip(x_L, x_R)])

        best_F = None
        min_error = np.Inf
        n_cols = len(A[-1])

        for i in range(n_cols):
            indices = list(range(n_cols))
            selected_column_index = indices.pop(i)
            U = A[:, indices]
            u = A[:, selected_column_index]

            f_prime = np.linalg.inv(U.T @ U) @ U.T @ u
            F = -np.r_[f_prime, -1.0].reshape(3, 3)

            residual = Fundamental._compute_algebraic_residual(F, x_L, x_R)
            error = np.linalg.norm(residual)

            if error < min_error:
                min_error = error
                best_F = F

        return best_F

    @staticmethod
    def _eigen_analysis(*, x_L: np.ndarray, x_R: np.ndarray):
        """
        See 3.2.2 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        # Re-write x_L.T @ F @ x_R = 0 to A @ f = 0
        A = np.asarray([np.kron(p_L, p_R) for (p_L, p_R) in zip(x_L, x_R)])
        _, eigenvectors = np.linalg.eig(A.T @ A)
        F_LR = eigenvectors[:, -1].reshape(3, 3)
        return F_LR

    @staticmethod
    def _impose_F_rank(F_LR: np.ndarray):
        """
        Because of the existence of skewed symmetric matrix related to translation vector in F,
        the rank is at most 2, we need to impose it.

        See 3.2.3 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        U, singular_values, Vt = np.linalg.svd(F_LR)
        F_LR = U @ np.diag([singular_values[0], singular_values[1], 0.0]) @ Vt
        return F_LR

    @staticmethod
    def _initialze(*, x_L: np.ndarray, x_R: np.ndarray) -> np.ndarray:
        """Initialize a initial F estimation for later optimization."""
        try:
            F_LR = Fundamental._eigen_analysis(x_L=x_L, x_R=x_R)
        except Exception as e:
            print(f"Use eight point due to SVD error: {e}")
            F_LR = Fundamental._linear_least_square_eight_point(x_L=x_L, x_R=x_R)
        return F_LR

    @staticmethod
    def _optimize(*, x_L: np.ndarray, x_R: np.ndarray, initial_F_LR: np.ndarray) -> np.ndarray:
        """Optimization the fundamental matrix.

        Optimization by exploiting the epipolar constraints and
        minimizing the distances from points to their conjugate epipolar lines.

        See 3.4 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        try:
            result = least_squares(
                fun=Fundamental._residual,
                x0=initial_F_LR.reshape(-1),
                args=(x_L, x_R),
                loss="cauchy",
            )

            if not result["success"]:
                print("Optimization failed! Use initial F.")
                return initial_F_LR

            F_LR = result["x"].reshape(3, 3)
            F_LR = Fundamental._impose_F_rank(F_LR)
            return F_LR

        except Exception as e:
            print(f"Optimization failed, use initial F. Error: {e}")
            return initial_F_LR

    @staticmethod
    def get_epilines_L(*, x_R, F_LR):
        """
        Get epipolar lines on the left image from the right points.

        We have,

            x_L.T @ (F @ x_R) = 0 => lines = (F @ x_R).T = x_R.T @ F.T.

        Note that `x_R` are already row vectors.
        """
        return x_R @ F_LR.T

    @staticmethod
    def get_epilines_R(*, x_L, F_LR):
        """
        Get epipolar lines on the right image from the left points.

        We have

            (x_L.T @ F) @ x_R = 0 => lines = x_L.T @ F

        Note that `x_L` are already row vectors.
        """
        return x_L @ F_LR

    @staticmethod
    def _compute_distances_to_epilines(*, x, epilines):
        """Compute the distances of the points to their conjugate epipolar lines."""
        return np.sum(epilines * x, axis=1) / np.linalg.norm(epilines[:, :2], axis=1)

    @staticmethod
    def _residual(f: np.ndarray, x_L: np.ndarray, x_R: np.ndarray):
        """Computes the distances from points to their conjugate epipolar lines."""
        F_LR = f.reshape(3, 3)

        epilines_L = Fundamental.get_epilines_L(x_R=x_R, F_LR=F_LR)
        distances_L = Fundamental._compute_distances_to_epilines(x=x_L, epilines=epilines_L)

        epilines_R = Fundamental.get_epilines_R(x_L=x_L, F_LR=F_LR)
        distances_R = Fundamental._compute_distances_to_epilines(x=x_R, epilines=epilines_R)

        return np.r_[distances_L, distances_R]

    @staticmethod
    def _compute_algebraic_residual(F_LR, x_L, x_R):
        """Compute the algebraic residual."""
        # NOTE: Diagonal means to remove the cross terms.
        # NOTE: Copy is necessary because somehow the output array from np.diagonal is readonly.
        return np.diagonal(homogeneous(x_L) @ F_LR @ homogeneous(x_R).T).copy()

    @staticmethod
    def compute_geometric_rms(*, F_LR: np.ndarray, x_L: np.ndarray, x_R: np.ndarray):
        """Computes the directional distances from points to their conjugate epilines."""
        distances = Fundamental._residual(F_LR.reshape(-1), homogeneous(x_L), homogeneous(x_R))
        return sqrt((distances ** 2).mean())

    @staticmethod
    def plot_epipolar_lines(
        image_L: np.ndarray,
        image_R: np.ndarray,
        points_L: np.ndarray,
        points_R: np.ndarray,
        F_LR: np.ndarray,
    ):
        import matplotlib.pyplot as plt

        width = image_L.shape[1]
        height = image_L.shape[0]

        colors = np.random.random((len(points_L), 3)) * 0.7 + 0.1

        def _plot_line(lines):
            for i, l in enumerate(lines):
                points = basic.get_line_points_in_image(l, width, height)
                plt.plot(points[:, 0], points[:, 1], alpha=0.8, color=colors[i])

        plt.figure()

        plt.subplot(121)
        plt.title("Corresponding epilines of (R) in (L)")
        plt.imshow(image_L)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        left_lines = Fundamental.get_epilines_L(x_R=homogeneous(points_R), F_LR=F_LR)
        _plot_line(left_lines)
        plt.scatter(points_L[:, 0], points_L[:, 1], color=colors)

        plt.subplot(122)
        plt.title("Corresponding epilines of (L) in (R)")
        plt.imshow(image_R)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        right_lines = Fundamental.get_epilines_R(x_L=homogeneous(points_L), F_LR=F_LR)
        _plot_line(right_lines)
        plt.scatter(points_R[:, 0], points_R[:, 1], color=colors)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _assert_symbolic_A():
        """This function is used for manual validation."""
        import sympy as sym

        x_L = np.array(np.r_[sym.symbols("x1 y1"), 1.0]).reshape(-1, 1)
        x_R = np.array(np.r_[sym.symbols("x2 y2"), 1.0]).reshape(-1, 1)

        F = []
        for row in range(1, 4):
            for col in range(1, 4):
                F.append(sym.Symbol(f"f{row}{col}"))
        F = np.asarray(F).reshape(3, 3)

        eqn1 = sym.Matrix(x_L.T @ F @ x_R)
        eqn2 = sym.Matrix(x_R.T @ F.T @ x_L)

        A = np.kron(x_L, x_R)
        f = F.reshape(-1)
        Af = A.T @ f
        eqn3 = sym.Matrix(Af)

        subs = dict(
            x1=1,
            x2=2,
            y1=3,
            y2=4,
            f11=1,
            f12=2,
            f13=3,
            f21=4,
            f22=5,
            f23=6,
            f31=7,
            f32=8,
            f33=9,
        )

        assert eqn1.subs(subs) == eqn2.subs(subs)
        assert eqn3.subs(subs) == eqn1.subs(subs)


def decompose_essential_matrix(*, E_LR: np.ndarray):
    """Decompose essential matrix from (L) to (R)."""
    U, _, Vt = np.linalg.svd(E_LR)

    W1 = Rotation.from_euler("zyx", [pi / 2.0, 0, 0]).as_matrix()
    W2 = Rotation.from_euler("zyx", [-pi / 2.0, 0, 0]).as_matrix()

    # Since cross(t, E) = 0 which means t is the left singular vector
    # associated with the 0 singular value>
    t_LR = U[:, -1]

    # TODO(kun): Check the solution for R.
    R1_LR = U @ W1.T @ Vt
    R2_LR = U @ W2.T @ Vt

    assert np.allclose(t_LR @ R1_LR, t_LR @ R2_LR)

    return R1_LR, R2_LR, t_LR


def triangulate(P1, P2, point1, point2):
    """Use projection matrices and pixels coordinates to find out the 3D points.

    Projection matrix must be a 3x4 augmented matrix.

    TODO(kun): Replace OpenCV usage.
    """
    object_points = cv2.triangulatePoints(P1, P2, point1, point2)
    object_points /= object_points[-1]
    return object_points.reshape(-1)[:3]
