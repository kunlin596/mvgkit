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
    """This class implements methods for computing fundamental (as well as essential) matrix.

    The epipolar coplanarity constraint can be expressed as the equations below as

    Here, x_L and x_R are the corresponding pixel coordinates in the frame (L) and (R) respectively,
    K_L and L_R is the camera matrices. R_LR is the relative orientation of frame (R) expressed in frame (L),
    t_LR is the skew symmetric matrix of the translation vector t from frame (L) to (R).
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

        F_RL, inlier_mask = Fundamental._initialze(x_L=normalized_x_L, x_R=normalized_x_R)
        F_RL = Fundamental._optimize(
            x_L=normalized_x_L[inlier_mask], x_R=normalized_x_R[inlier_mask], initial_F_RL=F_RL
        )

        F_RL = N_R.T @ F_RL @ N_L
        F_RL = Fundamental._impose_F_rank(F_RL)
        F_RL /= F_RL[-1, -1]
        return F_RL, inlier_mask

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
        A = np.asarray([np.kron(p_R, p_L) for (p_L, p_R) in zip(x_L, x_R)])

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
        A = np.asarray([np.kron(p_R, p_L) for (p_L, p_R) in zip(x_L, x_R)])
        _, eigenvectors = np.linalg.eig(A.T @ A)
        F_RL = eigenvectors[:, -1].reshape(3, 3)
        return F_RL

    @staticmethod
    def _ransac_point_registrator(*, x_L: np.ndarray, x_R: np.ndarray, num_iter=1000, atol=0.01):
        i = 0
        max_num_inliers = -1
        best_F_RL = None
        best_inlier_mask = None
        num_points = len(x_L)
        while i < num_iter:
            samples = np.random.randint(0, num_points, 8)
            while len(set(samples)) != len(samples):
                samples = np.random.randint(0, num_points, 8)

            F_RL = Fundamental._eigen_analysis(x_L=x_L[samples], x_R=x_R[samples])
            distances = np.abs(Fundamental._residual(F_RL.reshape(-1), x_L, x_R))
            distances_L = distances[: len(distances) // 2]
            distances_R = distances[len(distances) // 2 :]

            inlier_mask = (distances_L < atol) & (distances_R < atol)
            num_inliers = np.count_nonzero(inlier_mask)

            if num_inliers > max_num_inliers:
                max_num_inliers = num_inliers
                best_F_RL = F_RL
                best_inlier_mask = inlier_mask
            i += 1

        return best_F_RL, best_inlier_mask

    @staticmethod
    def _impose_F_rank(F_RL: np.ndarray):
        """
        Because of the existence of skewed symmetric matrix related to translation vector in F,
        the rank is at most 2, we need to impose it.

        See 3.2.3 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        U, singular_values, Vt = np.linalg.svd(F_RL)
        F_RL = U @ np.diag([singular_values[0], singular_values[1], 0.0]) @ Vt
        return F_RL

    @staticmethod
    def _initialze(*, x_L: np.ndarray, x_R: np.ndarray) -> np.ndarray:
        """Initialize a initial F estimation for later optimization."""
        F_RL, inlier_mask = Fundamental._ransac_point_registrator(x_L=x_L, x_R=x_R)
        if F_RL is None:
            print("RANSAC failed! Use all points")
            F_RL = Fundamental._eigen_analysis(x_L=x_L, x_R=x_R)
            return F_RL, np.ones(len(x_L), dtype=np.uint8).view(bool)
        return F_RL, inlier_mask

    @staticmethod
    def _optimize(*, x_L: np.ndarray, x_R: np.ndarray, initial_F_RL: np.ndarray) -> np.ndarray:
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
                x0=initial_F_RL.reshape(-1),
                args=(x_L, x_R),
                loss="huber",
            )

            if not result["success"]:
                print("Optimization failed! Use initial F.")
                return initial_F_RL

            F_RL = result["x"].reshape(3, 3)
            F_RL = Fundamental._impose_F_rank(F_RL)
            return F_RL

        except Exception as e:
            print(f"Optimization failed, use initial F. Error: {e}")
            return initial_F_RL

    @staticmethod
    def get_epipole_L(*, F_RL):
        assert np.allclose(np.linalg.det(F_RL), 0.0)
        eigenvalues, eigenvectors = np.linalg.eig(F_RL)
        epipole_L = eigenvectors[:, eigenvalues.argmin()]
        epipole_L /= epipole_L[-1]
        return epipole_L

    @staticmethod
    def get_epipole_R(*, F_RL):
        assert np.allclose(np.linalg.det(F_RL), 0.0)
        eigenvalues, eigenvectors = np.linalg.eig(F_RL.T)
        epipole_R = eigenvectors[:, eigenvalues.argmin()]
        epipole_R /= epipole_R[-1]
        return epipole_R

    @staticmethod
    def get_epilines_L(*, x_R, F_RL):
        """Get epipolar lines on the left image from the right points."""
        return x_R @ F_RL

    @staticmethod
    def get_epilines_R(*, x_L, F_RL):
        """Get epipolar lines on the right image from the left points."""
        return x_L @ F_RL.T

    @staticmethod
    def _compute_distances_to_epilines(*, x, epilines):
        """Compute the distances of the points to their conjugate epipolar lines."""
        return np.sum(epilines * x, axis=1) / np.linalg.norm(epilines[:, :2], axis=1)

    @staticmethod
    def _residual(f: np.ndarray, x_L: np.ndarray, x_R: np.ndarray):
        """Computes the distances from points to their conjugate epipolar lines."""
        F_RL = f.reshape(3, 3)

        epilines_L = Fundamental.get_epilines_L(x_R=x_R, F_RL=F_RL)
        distances_L = Fundamental._compute_distances_to_epilines(x=x_L, epilines=epilines_L)

        epilines_R = Fundamental.get_epilines_R(x_L=x_L, F_RL=F_RL)
        distances_R = Fundamental._compute_distances_to_epilines(x=x_R, epilines=epilines_R)

        return np.r_[distances_L, distances_R]

    @staticmethod
    def _compute_algebraic_residual(F_RL, x_L, x_R):
        """Compute the algebraic residual."""
        # NOTE: Diagonal means to remove the cross terms.
        # NOTE: Copy is necessary because somehow the output array from np.diagonal is readonly.
        return np.diagonal(homogeneous(x_R) @ F_RL @ homogeneous(x_L).T).copy()

    @staticmethod
    def compute_geometric_rms(*, F_RL: np.ndarray, x_L: np.ndarray, x_R: np.ndarray):
        """Computes the directional distances from points to their conjugate epilines."""
        distances = Fundamental._residual(F_RL.reshape(-1), homogeneous(x_L), homogeneous(x_R))
        return sqrt((distances ** 2).mean())

    @staticmethod
    def plot_epipolar_lines(
        image_L: np.ndarray,
        image_R: np.ndarray,
        points_L: np.ndarray,
        points_R: np.ndarray,
        F_RL: np.ndarray,
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
        left_lines = Fundamental.get_epilines_L(x_R=homogeneous(points_R), F_RL=F_RL)
        _plot_line(left_lines)
        plt.scatter(points_L[:, 0], points_L[:, 1], color=colors)

        plt.subplot(122)
        plt.title("Corresponding epilines of (L) in (R)")
        plt.imshow(image_R)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        right_lines = Fundamental.get_epilines_R(x_L=homogeneous(points_L), F_RL=F_RL)
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

        F_RL = []
        for row in range(1, 4):
            for col in range(1, 4):
                F_RL.append(sym.Symbol(f"f{row}{col}"))
        F_RL = np.asarray(F_RL).reshape(3, 3)

        eqn1 = sym.Matrix(x_R.T @ F_RL @ x_L)
        eqn2 = sym.Matrix(x_L.T @ F_RL.T @ x_R)

        A = np.kron(x_R, x_L)
        f = F_RL.reshape(-1)
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


def decompose_essential_matrix(*, E_RL: np.ndarray):
    """Decompose essential matrix describing the change of frame from (L) to (R)."""
    U, _, Vt = np.linalg.svd(E_RL)

    W = Rotation.from_euler("zyx", [pi / 2.0, 0, 0]).as_matrix()

    # Since cross(t, E) = 0 which means t is the left singular vector
    # associated with the 0 singular value>
    t_R = U[:, -1]

    R1_RL = U @ W @ Vt
    R2_RL = U @ W.T @ Vt

    # TODO(kun): Check why these cases can occur
    if np.linalg.det(R1_RL) < 0.0:
        R1_RL = -R1_RL

    if np.linalg.det(R2_RL) < 0.0:
        R2_RL = -R2_RL

    assert np.allclose(t_R @ R1_RL, t_R @ R2_RL)
    return R1_RL, R2_RL, t_R


def triangulate(P1, P2, points1, points2):
    """Use projection matrices and pixels coordinates to find out the 3D points.

    Projection matrix must be a 3x4 augmented matrix.

    TODO(kun): Replace OpenCV usage.
    """
    object_points = cv2.triangulatePoints(P1, P2, points1.T, points2.T).T
    object_points /= object_points[:, -1].reshape(-1, 1)
    return object_points[:, :3]
