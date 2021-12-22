#!/usr/bin/env python3


from math import sqrt
import numpy as np
from mvg.basic import get_isotropic_scaling_matrix_2d, homogeneous
from scipy.optimize import minimize


class Fundamental:
    @staticmethod
    def compute(*, x: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Solve F for x2.T @ F @ x = 0. Expressing x2 in x's frame."""
        assert len(x) == len(x2)

        N_x = get_isotropic_scaling_matrix_2d(x)
        N_x2 = get_isotropic_scaling_matrix_2d(x2)

        normalized_x = x @ N_x[:2, :2].T + N_x[:2, 2]
        normalized_x2 = x2 @ N_x2[:2, :2].T + N_x2[:2, 2]

        F = Fundamental._initialze(x=normalized_x, x2=normalized_x2)
        F = Fundamental._optimize(x=normalized_x, x2=normalized_x2, initial_F=F)

        F = N_x2.T @ F @ N_x
        F /= F[-1, -1]
        return F

    @staticmethod
    def _make_observation_matrix(*, x: np.ndarray, x2: np.ndarray):
        """
        Creating the A matrix in the formula rewriting,
        x2.T @ F @ x = 0 => A @ f = 0 for homogeneous linear equations solving.
        """
        A = np.vstack(
            [
                x2[:, 0] * x[:, 0],
                x2[:, 0] * x[:, 1],
                x2[:, 0],
                x2[:, 1] * x[:, 0],
                x2[:, 1] * x[:, 1],
                x2[:, 1],
                x[:, 0],
                x[:, 1],
                np.ones(len(x[:, 0])),
            ]
        ).T
        return A

    @staticmethod
    def _linear_least_square_eight_point(*, x: np.ndarray, x2: np.ndarray):
        """Solve homogeneous linear equations

        Since the objective function is A @ f = 0, w/o loss of generality,
        we can choose to set one of f to be 0, and solve the linear system.

        However, since we don't know which entry of f is non-zero,
        we need to test it 9 times.

        See 3.2.1 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        A = Fundamental._make_observation_matrix(x=x, x2=x2)

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

            residual = Fundamental._compute_algebraic_residual(F, x, x2)
            error = np.linalg.norm(residual)

            if error < min_error:
                min_error = error
                best_F = F

        best_F = Fundamental._impose_F_rank(best_F)
        return best_F

    @staticmethod
    def _eigen_analysis(*, x: np.ndarray, x2: np.ndarray):
        """
        See 3.2.2 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        A = Fundamental._make_observation_matrix(x=x, x2=x2)
        # Pick the right singular vector with the smallest singular value
        _, _, vt = np.linalg.svd(A)
        F = vt[-1].reshape(3, 3)
        F = Fundamental._impose_F_rank(F)
        return F

    @staticmethod
    def _impose_F_rank(F: np.ndarray):
        """
        Because of the existence of skewed symmetric matrix related to translation vector in F,
        the rank is at most 2, we need to impose it.

        See 3.2.3 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        u, sig, vt = np.linalg.svd(F)
        F = u @ np.diag([sig[0], sig[1], 0.0]) @ vt
        F /= F[-1, -1]
        return F

    @staticmethod
    def _initialze(*, x: np.ndarray, x2: np.ndarray) -> np.ndarray:
        try:
            F = Fundamental._eigen_analysis(x=x, x2=x2)

            # This method should yield the same result as the method above
            # F2 = Fundamental._linear_least_square_eight_point(x=x, x2=x2)
            # assert np.allclose(F, F2)
        except Exception as e:
            print(f"Use eight point due to SVD error: {e}")
            F = Fundamental._linear_least_square_eight_point(x=x, x2=x2)

        return F

    @staticmethod
    def _optimize(*, x: np.ndarray, x2: np.ndarray, initial_F: np.ndarray) -> np.ndarray:
        """Optimization using epipolar constraints

        See 3.4 of the paper below.
        Z. Zhang, “Determining the Epipolar Geometry and Its Uncertainty: A Review,”
        Technical Report RR-2927, INRIA, 1996.
        """
        try:
            # TODO: Check Nelder-Mead method
            result = minimize(
                fun=Fundamental._objective_func,
                x0=initial_F.reshape(-1),
                args=(homogeneous(x), homogeneous(x2)),
                method="Nelder-Mead",
                tol=0.1,
            )

            if not result["success"]:
                print("Optimization failed! Use initial F.")
                return initial_F

            F = result["x"].reshape(3, 3)
            F = Fundamental._impose_F_rank(F)
            return F

        except Exception as e:
            print(f"Optimization failed, use initial F. Error: {e}")
            return initial_F

    @staticmethod
    def get_left_epilines(*, x2, F):
        return x2 @ F

    @staticmethod
    def get_right_epilines(*, x, F):
        return (F @ x.T).T

    @staticmethod
    def _compute_distances_to_epilines(*, x, epilines):
        """
        Compute the squared sum of distances of the points
        to the epipolar lines computed from their corresponding points in another image.

        NOTE: the distances are signed distance, square it to remove the sign.
        """
        return (np.sum(epilines * x, axis=1) / np.linalg.norm(epilines[:, :2])) ** 2

    @staticmethod
    def _objective_func(f: np.ndarray, x: np.ndarray, x2: np.ndarray):
        F = f.reshape(3, 3)
        distances1 = Fundamental._compute_distances_to_epilines(x=x, epilines=x2 @ F)
        distances2 = Fundamental._compute_distances_to_epilines(x=x2, epilines=(F @ x.T).T)
        rms = sqrt((np.sum(distances1) + np.sum(distances2)) / len(x))
        return rms

    @staticmethod
    def _compute_algebraic_residual(F, x, x2):
        """Compute the algebraic residual"""
        # NOTE: Diagonal means to remove the cross terms.
        # NOTE: Copy is necessary because somehow the output array from np.diagonal is readonly.
        return np.diagonal(homogeneous(x2) @ F @ homogeneous(x).T).copy()

    @staticmethod
    def compute_rms(*, F: np.ndarray, x: np.ndarray, x2: np.ndarray):
        return Fundamental._objective_func(f=F.reshape(-1), x=homogeneous(x), x2=homogeneous(x2))

    @staticmethod
    def plot_epipolar_lines(
        image1: np.ndarray,
        image2: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray,
        F: np.ndarray,
    ):
        import matplotlib.pyplot as plt

        width = image1.shape[1]
        height = image1.shape[0]

        colors = np.random.random((len(points1), 3)) * 0.5 + 0.2

        def _plot_line(lines):
            for i, l in enumerate(lines):
                x = np.arange(0, width, 0.1)
                y = (-l[2] - x * l[0]) / l[1]
                valid = (0 <= y) & (y < height)
                plt.plot(x[valid], y[valid], alpha=0.8, color=colors[i])

        plt.figure()
        plt.subplot(221)
        plt.imshow(image1)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        plt.scatter(points1[:, 0], points1[:, 1], color=colors)
        left_lines = Fundamental.get_left_epilines(x2=homogeneous(points2), F=F)
        _plot_line(left_lines)

        plt.subplot(222)
        plt.imshow(image2)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        plt.scatter(points2[:, 0], points2[:, 1], color=colors)

        plt.subplot(223)
        plt.imshow(image1)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        plt.scatter(points1[:, 0], points1[:, 1], color=colors)

        plt.subplot(224)
        plt.imshow(image2)
        plt.xlim([0, width])
        plt.ylim([height, 0])
        plt.scatter(points2[:, 0], points2[:, 1], color=colors)
        right_lines = Fundamental.get_right_epilines(x=homogeneous(points1), F=F)
        _plot_line(right_lines)
        plt.show()
