#!/usr/bin/env python3

import numpy as np
from scipy.sparse.lil import lil_matrix

from mvg.basic import get_normalization_matrix_2d, homogeneous
from scipy.optimize import least_squares


class Homography:
    @staticmethod
    def compute(*, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute Homography defined in equation x_1 = H @ _0."""
        """Solve H for dst = H @ src"""
        assert len(src) == len(dst)
        N_src = get_normalization_matrix_2d(src)
        N_dst = get_normalization_matrix_2d(dst)

        normalized_src = src @ N_src[:2, :2] + N_src[:2, 2]
        normalized_dst = dst @ N_dst[:2, :2] + N_dst[:2, 2]

        H = Homography._initialze(src=normalized_src, dst=normalized_dst)
        H = Homography._refine(src=normalized_src, dst=normalized_dst, initial_H=H)

        # Un-normalize H
        H = np.linalg.inv(N_dst) @ H @ N_src
        return H

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
                fun=Homography._residual,
                x0=initial_H.reshape(-1),
                jac_sparsity=Homography._get_sparsity(len(src)),
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
