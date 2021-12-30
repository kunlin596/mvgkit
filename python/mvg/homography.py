#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
from scipy.sparse.lil import lil_matrix
from scipy.optimize import least_squares

from mvg.basic import get_isotropic_scaling_matrix_2d, homogeneous


@dataclass
class Homography2d:
    """This class defines a homography in P2 space, i.e, 2D homogeneous coordinate system."""

    H: np.ndarray

    def __post_init__(self):
        Homography2d.ensure_projectivity(self.H)
        self.H /= self.H[-1, -1]

    def decompose_SAP(self):
        """Decompose a homography into a chain of transformations.

        This chain from left to right includes similarity, affine, projective transformation.

            H_sap = H_s @ H_a @ H_p

        See Multiple View Geometry 2.4.6.
        """
        H_sap = self.H
        H_p = np.eye(3)
        H_p[-1] = H_sap[-1]

        H_sa = H_sap @ np.linalg.inv(H_p)

        H_s_rect22, H_a_rect22 = np.linalg.qr(H_sa[:2, :2])

        H_s = np.eye(3)
        H_s[:2, :2] = H_s_rect22
        H_s[:2, 2] = H_sa[:2, 2]

        H_a = np.eye(3)
        H_a[:2, :2] = H_a_rect22

        assert np.allclose(H_s @ H_a @ H_p, H_sap)
        return (H_s, H_a, H_p)

    @staticmethod
    def from_matrix(H: np.ndarray):
        assert H.shape == (3, 3)
        return Homography2d(H.copy())

    @staticmethod
    def from_array(h: np.ndarray):
        assert len(h) == 9
        return Homography2d(h.copy().reshape(3, 3))

    def as_matrix(self):
        return self.H

    def as_array(self):
        return self.H.reshape(-1)

    @staticmethod
    def ensure_projectivity(H: np.ndarray):
        assert H.shape == (3, 3), "H is not a square 3 x 3 matrix!"
        # assert np.linalg.det(H) > 0, "H is not invertible!"

    def transform(self, points: np.ndarray):
        assert len(points.shape) == 2

        is_homogeneous = True
        if points.shape[1] == 2:
            is_homogeneous = False
            points = homogeneous(points)

        transformed = points @ self.as_matrix().T
        transformed /= transformed[:, -1].reshape(-1, 1)

        if not is_homogeneous:
            transformed = transformed[:, :2]

        return transformed


class HomographySolver2d:
    @staticmethod
    def compute(*, src: np.ndarray, dst: np.ndarray) -> Homography2d:
        """Solve H for dst = H @ src"""
        assert len(src) == len(dst)
        N_src = get_isotropic_scaling_matrix_2d(src)
        N_dst = get_isotropic_scaling_matrix_2d(dst)

        normalized_src = src @ N_src[:2, :2] + N_src[:2, 2]
        normalized_dst = dst @ N_dst[:2, :2] + N_dst[:2, 2]

        H = HomographySolver2d._initialze(src=normalized_src, dst=normalized_dst)
        H = HomographySolver2d._refine(src=normalized_src, dst=normalized_dst, initial_H=H)

        # Un-normalize H
        H = np.linalg.inv(N_dst) @ H @ N_src
        return Homography2d(H)

    @staticmethod
    def _initialze(*, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        A = []
        for i in range(len(src)):
            srcp = src[i]
            dstp = dst[i]
            A.append(
                [
                    -srcp[0],
                    -srcp[1],
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    dstp[0] * srcp[0],
                    dstp[0] * srcp[1],
                    dstp[0],
                ]
            )
            A.append(
                [
                    0.0,
                    0.0,
                    0.0,
                    -srcp[0],
                    -srcp[1],
                    -1.0,
                    dstp[1] * srcp[0],
                    dstp[1] * srcp[1],
                    dstp[1],
                ]
            )
        A = np.asarray(A)

        _, _, vt = np.linalg.svd(A)
        H = vt[-1].reshape(3, 3) / vt[-1, -1]
        return H

    @staticmethod
    def _get_sparsity(n_point_pairs: int):
        J = lil_matrix((2 * n_point_pairs, 9), dtype=np.int32)
        J[::2, 3:6] = 1
        J[1::2, :3] = 1
        return J

    @staticmethod
    def _refine(*, src: np.ndarray, dst: np.ndarray, initial_H: np.ndarray) -> np.ndarray:
        try:
            result = least_squares(
                fun=HomographySolver2d._residual,
                x0=initial_H.reshape(-1),
                jac_sparsity=HomographySolver2d._get_sparsity(len(src)),
                args=(src, dst),
            )

            if not result["success"]:
                return initial_H

            H = result["x"].reshape(3, 3)
            H /= H[-1, -1]
            return H

        except Exception as e:
            print(e)
            return initial_H

    @staticmethod
    def _residual(h: np.ndarray, src: np.ndarray, dst: np.ndarray):
        src = homogeneous(src)
        transformed_src = src @ h.reshape(3, 3).T
        transformed_src[:, :2] /= transformed_src[:, -1].reshape(-1, 1)
        transformed_src = transformed_src[:, :2]
        return dst.reshape(-1) - transformed_src.reshape(-1)
