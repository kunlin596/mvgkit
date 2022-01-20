#!/usr/bin/env python3

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from mvg.basic import SE3
from mvg.camera import CameraMatrix, project_points
from mvg.dataloader import ImageFileReader
from mvg.features import SIFT, Matcher
from mvg.image_processing import Image
from mvg.stereo import Fundamental, decompose_essential_matrix, triangulate
from scipy.optimize import least_squares


class SFM:
    """Incremental SfM

    References:

        Snavely, N., Seitz, S.M. and Szeliski, R., 2006.
        Photo tourism: exploring photo collections in 3D.
        In ACM siggraph 2006 papers (pp. 835-846).

        Snavely, N., Seitz, S.M. and Szeliski, R., 2008.
        Modeling the world from internet photo collections.
        International journal of computer vision, 80(2), pp.189-210.

    """

    @dataclass
    class _ImageCorrespondencePack:
        image1: np.ndarray
        image2: np.ndarray
        raw_keypoints1: np.ndarray
        raw_keypoints2: np.ndarray
        descriptors1: np.ndarray
        descriptors2: np.ndarray
        matched_keypoints1: np.ndarray
        matched_keypoints2: np.ndarray

        # TODO(kun): add intrinsic parameter extractor
        camera_matrix1: Optional[CameraMatrix]
        camera_matrix2: Optional[CameraMatrix]

    _data: pd.DataFrame

    def __init__(self) -> None:
        pass

    @staticmethod
    def _compute_correspondence(image1, image2) -> _ImageCorrespondencePack:
        keypoints1, descriptors1 = SIFT.detect(image1)
        keypoints2, descriptors2 = SIFT.detect(image2)
        matches = Matcher.match(descriptors1=descriptors1, descriptors2=descriptors2)
        points1, points2, _ = Matcher.get_matched_points(
            keypoints1, keypoints2, matches, dist_threshold=0.8
        )

        return SFM._ImageCorrespondencePack(
            image1=image1,
            image2=image2,
            raw_keypoints1=keypoints1,
            raw_keypoints2=keypoints2,
            descriptors1=descriptors1,
            descriptors2=descriptors2,
            matched_keypoints1=points1,
            matched_keypoints2=points2,
            camera_matrix1=None,
            camera_matrix2=None,
        )

    @staticmethod
    def _get_correct_transformation(F_RL, K_L, K_R, points_L, points_R):
        E_RL = K_L.T @ F_RL @ K_R
        R1_RL, R2_RL, t_R = decompose_essential_matrix(E_RL=E_RL)

        P1 = K_L @ SE3.from_rotmat_tvec(np.eye(3), np.zeros(3)).as_augmented_matrix()

        T_RL_candidates = [
            SE3.from_rotmat_tvec(R1_RL, t_R),
            SE3.from_rotmat_tvec(R1_RL, -t_R),
            SE3.from_rotmat_tvec(R2_RL, t_R),
            SE3.from_rotmat_tvec(R2_RL, -t_R),
        ]

        P2_candidates = [K_L @ T_RL.as_augmented_matrix() for T_RL in T_RL_candidates]

        max_num_valid_points = -1

        best_T = None
        best_points3d_L = None

        for i, P2 in enumerate(P2_candidates):
            points3d_L = triangulate(P1, P2, points_L, points_R)

            inlier_mask = points3d_L[:, 2] > 1.0
            num_valid_points = np.count_nonzero(inlier_mask)

            if num_valid_points > max_num_valid_points:
                max_num_valid_points = num_valid_points

                best_T = T_RL_candidates[i]
                best_points3d_L = points3d_L

        return best_T, best_points3d_L

    @staticmethod
    def _compute_T_from_focal_length(x, F_RL, points_L, points_R, image1, image2):
        f1 = x[0]
        f2 = x[1]
        camera_matrix_L = CameraMatrix(
            fx=f1, fy=f1, cx=image1.data.shape[1] / 2.0, cy=image1.data.shape[0], s=0.0
        )
        camera_matrix_R = CameraMatrix(
            fx=f2, fy=f2, cx=image2.data.shape[1] / 2.0, cy=image2.data.shape[0], s=0.0
        )

        T_RL, points3d_L = SFM._get_correct_transformation(
            F_RL, camera_matrix_L.as_matrix(), camera_matrix_R.as_matrix(), points_L, points_R
        )
        return T_RL, points3d_L, camera_matrix_L, camera_matrix_R

    @staticmethod
    def _residual(x, F_RL, points_L, points_R, image1, image2):
        T_RL, points3d_L, camera_matrix_L, camera_matrix_R = SFM._compute_T_from_focal_length(
            x, F_RL, points_L, points_R, image1, image2
        )

        reprojected_L = project_points(object_points_W=points3d_L, camera_matrix=camera_matrix_L)
        reprojected_R = project_points(
            object_points_W=points3d_L, camera_matrix=camera_matrix_R, T_CW=T_RL
        )

        error1 = points_L - reprojected_L
        error2 = points_R - reprojected_R
        return np.r_[error1, error2].reshape(-1)

    @staticmethod
    def _optimize_focal_lengths(
        image1: Image, image2: Image, F_RL: np.ndarray, points_L: np.ndarray, points_R: np.ndarray
    ):

        best_result = None
        scores = []
        for x0 in range(100, 2000, 10):
            result = least_squares(
                x0=[x0, x0],
                fun=SFM._residual,
                args=(F_RL, points_L, points_R, image1.data, image2.data),
            )

            print(x0, result["optimality"])
            scores.append(result["optimality"])
            if best_result is None or (
                result["success"] and (result["optimality"] < best_result["optimality"])
            ):
                best_result = result

        T_RL, points3d_L, camera_matrix_L, camera_matrix_R = SFM._compute_T_from_focal_length(
            best_result["x"], F_RL, points_L, points_R, image1.data, image2.data
        )
        print(best_result)
        from IPython import embed

        embed()
        return T_RL, points3d_L, camera_matrix_L, camera_matrix_R

    @staticmethod
    def _compute_transformation(image1: Image, image2: Image):
        correspondence = SFM._compute_correspondence(image1.data, image2.data)
        F_RL, inlier_mask = Fundamental.compute(
            x_L=correspondence.matched_keypoints1, x_R=correspondence.matched_keypoints2
        )

        points_inliers_L = correspondence.matched_keypoints1[inlier_mask]
        points_inliers_R = correspondence.matched_keypoints2[inlier_mask]

        T_RL, points3d_L, camera_matrix_L, camera_matrix_R = SFM._optimize_focal_lengths(
            image1, image2, F_RL, points_inliers_L, points_inliers_R
        )

        reprojected1 = camera_matrix_L.project(points_C=points3d_L)
        reprojected2 = camera_matrix_R.project(points_C=points3d_L @ T_RL.R.as_matrix().T + T_RL.t)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(121)
        plt.imshow(image1.data)
        plt.scatter(points_inliers_L[:, 0], points_inliers_L[:, 1], alpha=0.5, c="r")
        plt.scatter(reprojected1[:, 0], reprojected1[:, 1], alpha=0.5, c="g")

        plt.subplot(122)
        plt.imshow(image2.data)
        plt.scatter(points_inliers_R[:, 0], points_inliers_R[:, 1], alpha=0.5, c="r")
        plt.scatter(reprojected2[:, 0], reprojected2[:, 1], alpha=0.5, c="g")
        plt.show()

        from IPython import embed

        embed()

        return T_RL, points3d_L

    def initialize(self, image1: Image, image2: Image):
        """Compute initial camera pose"""
        T_RL, points_3d = self._compute_transformation(image1, image2)
        # TODO

    def _find_closest_image(self, image: Image):
        return image

    def add_image(self, new_image: Image):
        """Incrementally add image into collections"""
        closest_image = self._find_closest_image(new_image)
        T_RL, points_3d = self._compute_transformation(closest_image, new_image)
        # TODO

    def optimize(self, image_indices: Optional[np.ndarray] = None):
        """Optimize all parameters using image indices, if not given, optimize all."""
        # TODO
        pass

    @staticmethod
    def run_all(datapath: Path):
        """Run SfM"""
        reader = ImageFileReader(datapath)

        with open(datapath / "meta.json") as f:
            filemeta = json.load(f)

        image1 = reader.readone(datapath / filemeta["initialization"][0])
        image2 = reader.readone(datapath / filemeta["initialization"][1])

        sfm = SFM()
        sfm.initialize(image1, image2)

        for image in reader.read():
            sfm.add_image(image)
            sfm.optimize()


if __name__ == "__main__":
    p = Path("/Users/linkun/Developer/data/NotreDame/images")
    SFM.run_all(p)
