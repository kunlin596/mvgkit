#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
from mvg.basic import SE3
from mvg.camera import CameraMatrix
from mvg.features import SIFT, Matcher
from mvg.stereo import Fundamental, decompose_essential_matrix, triangulate
import numpy as np
import pandas as pd
from dataclasses import dataclass

from pandas.core.frame import DataFrame

from build.lib.mvg.dataloader import ImageFileReader


class SFM:
    """Incremental SfM

    References:

        Snavely, N., Seitz, S.M. and Szeliski, R., 2006.
        Photo tourism: exploring photo collections in 3D.
        In ACM siggraph 2006 papers (pp. 835-846).

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
    def _get_correct_transformation(R1_RL, R2_RL, t_R, K, points_L, points_R):
        P1 = K @ SE3.from_rotmat_tvec(np.eye(3), np.zeros(3)).as_augmented_matrix()

        T_RL_candidates = [
            SE3.from_rotmat_tvec(R1_RL, t_R),
            SE3.from_rotmat_tvec(R1_RL, -t_R),
            SE3.from_rotmat_tvec(R2_RL, t_R),
            SE3.from_rotmat_tvec(R2_RL, -t_R),
        ]

        P2_candidates = [K @ T_RL.as_augmented_matrix() for T_RL in T_RL_candidates]

        max_num_valid_points = -1

        best_T = None
        best_points_3d = None

        for i, P2 in enumerate(P2_candidates):
            points_3d = triangulate(P1, P2, points_L, points_R)

            inlier_mask = points_3d[:, 2] > 1.0
            num_valid_points = np.count_nonzero(inlier_mask)

            if num_valid_points > max_num_valid_points:
                max_num_valid_points = num_valid_points

                best_T = T_RL_candidates[i]
                best_points_3d = points_3d

        return best_T, best_points_3d

    @staticmethod
    def _compute_transformation(image1, image2):
        correspondence = SFM._compute_correspondence(image1, image2)
        F_RL, inlier_mask = Fundamental.compute(
            x_L=correspondence.matched_keypoints1, x_R=correspondence.matched_keypoints2
        )
        points_inliers_L = correspondence.matched_keypoints1[inlier_mask]
        points_inliers_R = correspondence.matched_keypoints2[inlier_mask]

        K_L = correspondence.camera_matrix1.as_matrix()
        K_R = correspondence.camera_matrix2.as_matrix()
        E_RL = K_L.T @ F_RL @ K_R
        R1_RL, R2_RL, t_R = decompose_essential_matrix(E_RL=E_RL)

        # NOTE: Passing K_L since the 3d points are assumed to be in frame (L).
        T_RL, points_3d = SFM._get_correct_transformation(
            R1_RL, R2_RL, t_R, K_L, points_inliers_L, points_inliers_R
        )
        return T_RL, points_3d

    def _initialize(self, image1, image2):
        """Compute initial camera pose"""
        T_RL, points_3d = self._compute_transformation(image1, image2)
        # TODO

    def _find_closest_image(self, image):
        pass

    def _add_image(self, new_image):
        """Incrementally add image into collections"""
        closest_image = self._find_closest_image(new_image)
        T_RL, points_3d = self._compute_transformation(closest_image, new_image)
        # TODO

    def _optimize(self, image_indices: Optional[np.ndarray] = None):
        """Optimize all parameters using image indices, if not given, optimize all."""
        # TODO
        pass

    def run_all(self, datapath: Path):
        """Run SfM"""
        reader = ImageFileReader(datapath)
        image1 = reader.read()
        image2 = reader.read()
        self._initialize(image1, image2)
        for image in reader.read():
            self._add_image(image)
            self._optimize()


if __name__ == "__main__":
    pass
