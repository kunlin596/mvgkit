#!/usr/bin/env python3
"""This module implement stereo vision related algorithms."""


from math import pi, sqrt
from dataclasses import dataclass
from typing import Optional

import cv2
from itertools import product

from scipy.optimize import least_squares
from scipy.spatial.transform.rotation import Rotation

import numpy as np

from mvg.homography import Homography2d
from mvg.basic import (
    SkewSymmetricMatrix3d,
    get_isotropic_scaling_matrix_2d,
    homogenize,
    get_line_points_in_image,
)


class Fundamental:
    """This class implements methods for computing fundamental (as well as essential) matrix.

    The epipolar coplanarity constraint can be expressed as the equations below as

    Here, x_L and x_R are the corresponding pixel coordinates in the frame (L) and (R) respectively,
    K_L and L_R is the camera matrices. R_LR is the relative orientation of frame (R) expressed in frame (L),
    t_LR is the skew symmetric matrix of the translation vector t from frame (L) to (R).
    """

    @dataclass
    class Options:
        num_iters: int = 1000
        atol: float = 0.01

    @staticmethod
    def compute(
        *, x_L: np.ndarray, x_R: np.ndarray, options: Optional[Options] = None
    ) -> np.ndarray:
        assert (len(x_L) >= 8) and (
            len(x_R) >= 8
        ), f"Not enough points! len(x_L): {len(x_L)}, len(x_R): {len(x_R)}."

        if options is None:
            options = Fundamental.Options()

        num_iters = options.num_iters
        atol = options.atol

        N_L = get_isotropic_scaling_matrix_2d(x_L)
        N_R = get_isotropic_scaling_matrix_2d(x_R)

        hom_x_L = homogenize(x_L)
        hom_x_R = homogenize(x_R)

        normalized_x_L = hom_x_L @ N_L.T
        normalized_x_R = hom_x_R @ N_R.T

        F_RL, inlier_mask = Fundamental._initialze(
            x_L=normalized_x_L, x_R=normalized_x_R, num_iters=num_iters, atol=atol
        )
        F_RL = Fundamental._optimize(
            x_L=normalized_x_L[inlier_mask], x_R=normalized_x_R[inlier_mask], initial_F_RL=F_RL
        )

        F_RL = N_R.T @ F_RL @ N_L
        F_RL = Fundamental._impose_F_rank(F_RL)
        F_RL /= F_RL[-1, -1]

        return F_RL, inlier_mask

    @staticmethod
    def _linear_least_square_eight_point(*, x_L: np.ndarray, x_R: np.ndarray):
        """Solve homogenize linear equations

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
        _, _, vt = np.linalg.svd(A)
        F_RL = vt[-1].reshape(3, 3)
        return F_RL

    @staticmethod
    def _ransac_point_registrator(*, x_L: np.ndarray, x_R: np.ndarray, num_iters=1000, atol=0.01):
        i = 0
        max_num_inliers = -1
        best_F_RL = None
        best_inlier_mask = None
        num_points = len(x_L)
        while i < num_iters:
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
    def _initialze(*, x_L: np.ndarray, x_R: np.ndarray, num_iters: int, atol: float) -> np.ndarray:
        """Initialize a initial F estimation for later optimization."""
        assert len(x_L) >= 8, f"Number of points are is {len(x_L)} less than 8!"
        F_RL, inlier_mask = Fundamental._ransac_point_registrator(
            x_L=x_L, x_R=x_R, num_iters=num_iters, atol=atol
        )
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
            print("Optimizing F...")
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
        return np.real(epipole_L)

    @staticmethod
    def get_epipole_R(*, F_RL):
        assert np.allclose(np.linalg.det(F_RL), 0.0)
        eigenvalues, eigenvectors = np.linalg.eig(F_RL.T)
        epipole_R = eigenvectors[:, eigenvalues.argmin()]
        epipole_R /= epipole_R[-1]
        return np.real(epipole_R)

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
        return np.diagonal(homogenize(x_R) @ F_RL @ homogenize(x_L).T).copy()

    @staticmethod
    def compute_geometric_rms(*, F_RL: np.ndarray, x_L: np.ndarray, x_R: np.ndarray):
        """Computes the directional distances from points to their conjugate epilines."""
        distances = Fundamental._residual(F_RL.reshape(-1), homogenize(x_L), homogenize(x_R))
        return sqrt((distances**2).mean())

    @staticmethod
    def plot_epipolar_lines(
        image_L: np.ndarray,
        image_R: np.ndarray,
        points_L: np.ndarray,
        points_R: np.ndarray,
        F_RL: np.ndarray,
    ):
        import matplotlib.pyplot as plt

        colors = np.random.random((len(points_L), 3)) * 0.7 + 0.1

        def _plot_line(ax, lines, width, height):
            for i, l in enumerate(lines):
                points = get_line_points_in_image(l, width, height)
                ax.plot(points[:, 0], points[:, 1], alpha=0.8, color=colors[i])

        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

        width = image_L.shape[1]
        height = image_L.shape[0]

        ax1.set_title("Corresponding epilines of (R) in (L)")
        ax1.imshow(image_L)
        ax1.set_xlim([0, width])
        ax1.set_ylim([height, 0])
        left_lines = Fundamental.get_epilines_L(x_R=homogenize(points_R), F_RL=F_RL)
        _plot_line(ax1, left_lines, width, height)
        ax1.scatter(points_L[:, 0], points_L[:, 1], color=colors)

        width = image_R.shape[1]
        height = image_R.shape[0]

        ax2.set_title("Corresponding epilines of (L) in (R)")
        ax2.imshow(image_R)
        ax2.set_xlim([0, width])
        ax2.set_ylim([height, 0])
        right_lines = Fundamental.get_epilines_R(x_L=homogenize(points_L), F_RL=F_RL)
        _plot_line(ax2, right_lines, width, height)
        ax2.scatter(points_R[:, 0], points_R[:, 1], color=colors)

        plt.tight_layout()
        plt.show()

    # @staticmethod
    # def _assert_symbolic_A():
    #     """This function is used for manual validation."""
    #     import sympy as sym

    #     x_L = np.array(np.r_[sym.symbols("x1 y1"), 1.0]).reshape(-1, 1)
    #     x_R = np.array(np.r_[sym.symbols("x2 y2"), 1.0]).reshape(-1, 1)

    #     F_RL = []
    #     for row in range(1, 4):
    #         for col in range(1, 4):
    #             F_RL.append(sym.Symbol(f"f{row}{col}"))
    #     F_RL = np.asarray(F_RL).reshape(3, 3)

    #     eqn1 = sym.Matrix(x_R.T @ F_RL @ x_L)
    #     eqn2 = sym.Matrix(x_L.T @ F_RL.T @ x_R)

    #     A = np.kron(x_R, x_L)
    #     f = F_RL.reshape(-1)
    #     Af = A.T @ f
    #     eqn3 = sym.Matrix(Af)

    #     subs = dict(
    #         x1=1,
    #         x2=2,
    #         y1=3,
    #         y2=4,
    #         f11=1,
    #         f12=2,
    #         f13=3,
    #         f21=4,
    #         f22=5,
    #         f23=6,
    #         f31=7,
    #         f32=8,
    #         f33=9,
    #     )

    #     assert eqn1.subs(subs) == eqn2.subs(subs)
    #     assert eqn3.subs(subs) == eqn1.subs(subs)


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


class AffinityRecoverySolver:
    """This class implement the algorithm for computing affinity recovery homography."""

    @staticmethod
    def _compute_image_point_matrices(image_shape):
        # Eq. 11
        w = image_shape[1]
        h = image_shape[0]

        PPT = w * h / 12.0 * np.diag([w**2 - 1.0, h**2 - 1.0, 0.0])

        pc = np.asarray([(w - 1) / 2, (h - 1) / 2, 1.0]).reshape(-1, 1)
        pcpcT = pc @ pc.T

        return (PPT, pcpcT)

    @staticmethod
    def _compute_initial_guess(A, B):
        # 5.2
        L = np.linalg.cholesky(A)
        D = L.T
        D_inv = np.linalg.inv(D)
        eigenvalues, eigenvectors = np.linalg.eig(D_inv.T @ B @ D_inv)
        y = eigenvectors[:, eigenvalues.argmax()]
        return D_inv @ y

    @staticmethod
    def _residual_one_view(z, A, B):
        return ((z.T @ A @ z) / (z.T @ B @ z)).reshape(-1)

    @staticmethod
    def _residual_two_view(z, A_L, B_L, A_R, B_R):
        distortion_L = AffinityRecoverySolver._residual_one_view(z, A_L, B_L)
        distortion_R = AffinityRecoverySolver._residual_one_view(z, A_R, B_R)
        return np.r_[distortion_L, distortion_R]

    @staticmethod
    def _compute_z(A_L, B_L, A_R, B_R):
        # NOTE: z is 2-vector
        initial_z_L = AffinityRecoverySolver._compute_initial_guess(A_L, B_L)
        initial_z_R = AffinityRecoverySolver._compute_initial_guess(A_R, B_R)

        initial_z_L /= np.linalg.norm(initial_z_L)
        initial_z_R /= np.linalg.norm(initial_z_R)
        initial_z = (initial_z_L + initial_z_R) / 2.0

        z0 = initial_z / initial_z[initial_z.argmin()]
        result = least_squares(
            fun=AffinityRecoverySolver._residual_two_view,
            x0=z0,
            args=(A_L, B_L, A_R, B_R),
            loss="huber",
        )

        z = z0
        if result["success"]:
            z = result["x"]

        z = np.r_[z, 0.0]
        return z

    @staticmethod
    def _get_AB(mat, image_shape):
        PPT, pcpcT = AffinityRecoverySolver._compute_image_point_matrices(image_shape)
        A = mat.T @ PPT @ mat
        B = mat.T @ pcpcT @ mat
        return (A[:2, :2], B[:2, :2])

    @staticmethod
    def _build_projective_matrix(mat, z):
        w = mat @ z
        w /= w[-1]
        m = np.eye(3)
        m[-1] = w
        return m

    @staticmethod
    def _solve_one_view(mat1, mat2, image_shape_1, image_shape_2):

        A_L, B_L = AffinityRecoverySolver._get_AB(mat1, image_shape_1)
        A_R, B_R = AffinityRecoverySolver._get_AB(mat2, image_shape_2)

        # NOTE: z is 2-vector
        z = AffinityRecoverySolver._compute_z(A_L, B_L, A_R, B_R)

        Hp_L = AffinityRecoverySolver._build_projective_matrix(mat1, z)
        Hp_R = AffinityRecoverySolver._build_projective_matrix(mat2, z)

        return Hp_L, Hp_R

    @staticmethod
    def _solve(F_RL, image_shape_L, image_shape_R):
        """Solve the undistortion homography for left image."""
        epipole_L = Fundamental.get_epipole_L(F_RL=F_RL)
        e_x = SkewSymmetricMatrix3d.from_vec(epipole_L).as_matrix()

        A_L, B_L = AffinityRecoverySolver._get_AB(e_x, image_shape_L)
        A_R, B_R = AffinityRecoverySolver._get_AB(F_RL, image_shape_R)

        # NOTE: z is 2-vector
        z = AffinityRecoverySolver._compute_z(A_L, B_L, A_R, B_R)

        Hp_L = AffinityRecoverySolver._build_projective_matrix(e_x, z)
        Hp_R = AffinityRecoverySolver._build_projective_matrix(F_RL, z)

        return (Hp_L, Hp_R)

    @staticmethod
    def solve(*, F_RL, image_shape_L, image_shape_R):
        """Compute affinity recovery matrix from a image pair.

        This function assumes that the image size is identical and
        fundamental matrix is known and epipoles are computed as well.

        The minimal information required from image is only the width and height for each image

        The geometric distortion is defined as the distance from the image point
        to the center of the image,

            sum_{i-1}^n (w_i - w_C)^2 / w_c

        The idea is to measure the distance above for all of the pixels in both images.
        A key idea is to plug in the epipolar constraints into the objective function,
        such that we can optimize it accordingly. See derivation of equation 11.

        Once we found a ideal point z which can minimize the predefined distortion error,
        we can find out the where the epipolar line w at infinity.

        By building up the homography connecting the imaged epipolar line and
        the "moved" epipolar line we just derived, we can find out the desired homography.
        """

        return AffinityRecoverySolver._solve(F_RL, image_shape_L, image_shape_R)


class StereoRectifier:
    @staticmethod
    def _compute_projective_rectification(F_RL, image_shape_L, image_shape_R):
        return AffinityRecoverySolver.solve(
            F_RL=F_RL, image_shape_L=image_shape_L, image_shape_R=image_shape_R
        )

    @staticmethod
    def _compute_similarity_rectification(F_RL, Hp_L, Hp_R, image_shape_L, image_shape_R):
        points_L = np.asarray(list(product([0, image_shape_L[1]], [0, image_shape_L[0]])))[
            [0, 1, 3, 2]
        ]
        points_R = np.asarray(list(product([0, image_shape_R[1]], [0, image_shape_R[0]])))[
            [0, 1, 3, 2]
        ]

        warped_L = Homography2d.from_matrix(Hp_L).transform(points_L)
        warped_R = Homography2d.from_matrix(Hp_R).transform(points_R)

        vc = -min(warped_L[:, 1].min(), warped_R[:, 1].min())

        F13 = F_RL[0, 2]
        F23 = F_RL[1, 2]
        F31 = F_RL[2, 0]
        F32 = F_RL[2, 1]
        F33 = F_RL[2, 2]

        w_L = Hp_L[-1]
        wa = w_L[0]
        wb = w_L[1]

        Hr_L = np.asarray(
            [
                [F32 - wb * F33, wa * F33 - F31, 0.0],
                [F31 - wa * F33, F32 - wb * F33, F33 + vc],
                [0.0, 0.0, 1.0],
            ]
        )
        warped_L = warped_L @ Hr_L[:2, :2].T

        w_R = Hp_R[-1]
        wa = w_R[0]
        wb = w_R[1]
        Hr_R = np.asarray(
            [
                [wb * F33 - F23, F13 - wa * F33, 0.0],
                [wa * F33 - F13, wb * F33 - F23, vc],
                [0.0, 0.0, 1.0],
            ]
        )

        return (Hr_L, Hr_R)

    @staticmethod
    def _compute_shear_rectification(image_shape, H):
        w = image_shape[1] - 1.0
        h = image_shape[0] - 1.0

        half_w = w / 2.0
        half_h = h / 2.0

        a = H @ np.array([half_w, 0.0, 1.0])
        a /= a[-1]

        b = H @ np.array([w, half_h, 1.0])
        b /= b[-1]

        c = H @ np.array([half_w, h, 1.0])
        c /= c[-1]

        d = H @ np.array([0.0, half_h, 1.0])
        d /= d[-1]

        x = b - d
        y = c - a
        xu = x[0]
        xv = x[1]
        yu = y[0]
        yv = y[1]
        hw = h * w
        h2 = h**2
        w2 = w**2

        sa = (h2 * xv**2 + w2 * yv**2) / (hw * (xv * yu - xu * yv))
        sb = (h2 * xu * xv + w2 * yu * yv) / (hw * (xu * yv - xv * yu))

        Hs = np.eye(3)
        Hs[0, :2] = [sa, sb]
        if sa < 0.0:
            Hs[0, :2] = -Hs[0, :2]

        return Hs

    @staticmethod
    def _compute_scaling_factor(H, image_shape):
        points = np.asarray(list(product([0, image_shape[1]], [0, image_shape[0]])))[[0, 1, 3, 2]]
        warped_points = Homography2d.from_matrix(H).transform(points)

        area = cv2.contourArea(points.reshape(1, -1, 2).astype(np.float32))
        warped_area = cv2.contourArea(warped_points.reshape(1, -1, 2).astype(np.float32))

        scaling_factor = sqrt(area / warped_area)

        # Min corner before scaling, we cannot scale it just yet since the we need
        # to choose the best scaling factor for both images.
        min_corner = warped_points.min(axis=0)
        max_corner = warped_points.max(axis=0)
        return scaling_factor, min_corner, max_corner

    @staticmethod
    def _adjust_H_range(H_L, H_R, image_shape_L, image_shape_R):
        scaling_factor_L, min_corner_L, max_corner_L = StereoRectifier._compute_scaling_factor(
            H_L, image_shape_L
        )
        scaling_factor_R, min_corner_R, max_corner_R = StereoRectifier._compute_scaling_factor(
            H_R, image_shape_R
        )

        scaling_factor = max(scaling_factor_L, scaling_factor_R)

        scaled_min_corner = np.min(
            [min_corner_L * scaling_factor, min_corner_R * scaling_factor], axis=0
        )

        scaled_max_corner_L = max_corner_L * scaling_factor
        scaled_max_corner_R = max_corner_R * scaling_factor

        scale_mat = np.diag([scaling_factor, scaling_factor, 1.0])
        scale_mat[:2, 2] = -scaled_min_corner

        width_L, height_L = np.ceil(scaled_max_corner_L - scaled_min_corner).astype(np.int32)
        width_R, height_R = np.ceil(scaled_max_corner_R - scaled_min_corner).astype(np.int32)

        return (scale_mat @ H_L, scale_mat @ H_R, (width_L, height_L), (width_R, height_R))

    @staticmethod
    def _get_rectified_image_corners(H, image_shape):
        image_corners = np.asarray(list(product([0, image_shape[1]], [0, image_shape[0]])))[
            [0, 1, 3, 2]
        ]
        rectified_image_corners = Homography2d.from_matrix(H).transform(image_corners)
        return rectified_image_corners

    @staticmethod
    def compute_rectification_homography(
        image_L: np.ndarray, image_R: np.ndarray, F_RL: np.ndarray
    ):
        Hp_L, Hp_R = StereoRectifier._compute_projective_rectification(
            F_RL, image_L.shape, image_R.shape
        )

        Hr_L, Hr_R = StereoRectifier._compute_similarity_rectification(
            F_RL, Hp_L, Hp_R, image_L.shape, image_R.shape
        )

        Hs_L = StereoRectifier._compute_shear_rectification(image_L.shape, Hr_L @ Hp_L)
        Hs_R = StereoRectifier._compute_shear_rectification(image_R.shape, Hr_R @ Hp_R)

        H_L = Hs_L @ Hr_L @ Hp_L
        H_R = Hs_R @ Hr_R @ Hp_R

        H_L, H_R, size_L, size_R = StereoRectifier._adjust_H_range(
            H_L, H_R, image_L.shape, image_R.shape
        )

        rectified_image_corners_L = StereoRectifier._get_rectified_image_corners(H_L, image_L.shape)
        rectified_image_corners_R = StereoRectifier._get_rectified_image_corners(H_R, image_R.shape)

        return (H_L, H_R, size_L, size_R, rectified_image_corners_L, rectified_image_corners_R)


class StereoMatcher:
    @dataclass
    class Options:
        min_disparity: int = 0
        num_disparities: int = 50
        block_size: int = 3

    _F_RL: np.ndarray
    _H_L: np.ndarray
    _H_R: np.ndarray
    _image_size_L: np.ndarray
    _image_size_R: np.ndarray
    _rectified_size: np.ndarray
    _rectified_size_L: np.ndarray
    _rectified_size_R: np.ndarray
    _rectified_image_corners_L: np.ndarray
    _rectified_image_corners_R: np.ndarray
    _stereo_matcher: cv2.StereoSGBM
    _options: Options

    def __init__(self, F_RL, image_L, image_R, options: Optional[Options] = None):
        self._F_RL = F_RL
        if options is None:
            options = StereoMatcher.Options()

        self._image_size_L = (image_L.shape[1], image_L.shape[0])
        self._image_size_R = (image_R.shape[1], image_R.shape[0])

        self._options = options

        (
            self._H_L,
            self._H_R,
            self._rectified_size_L,
            self._rectified_size_R,
            self._rectified_image_corners_L,
            self._rectified_image_corners_R,
        ) = StereoRectifier.compute_rectification_homography(image_L, image_R, self._F_RL)

        self._rectified_size = tuple(
            np.max([self._rectified_size_L, self._rectified_size_R], axis=0)
        )

        assert len(self._rectified_image_corners_L) == len(self._rectified_image_corners_R)
        assert len(self._rectified_image_corners_L) == 4

        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=options.min_disparity,
            numDisparities=options.num_disparities,
            blockSize=options.block_size,
        )

    def compute(self, image_L, image_R):
        image_L = cv2.cvtColor(image_L, cv2.COLOR_RGB2GRAY)
        image_R = cv2.cvtColor(image_R, cv2.COLOR_RGB2GRAY)
        warped_L = cv2.warpPerspective(image_L, self._H_L, dsize=self._rectified_size)
        warped_R = cv2.warpPerspective(image_R, self._H_R, dsize=self._rectified_size)
        disparity_map = self._stereo_matcher.compute(warped_L, warped_R)
        return disparity_map
