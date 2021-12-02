#!/usr/bin/env python3

import numpy as np


class Homography:
    @staticmethod
    def compute(*, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute Homography defined in equation x_1 = H @ _0."""
        """Solve H for dst = H @ src"""
        assert len(src) == len(dst)
        H = Homography._initialze(src=src, dst=dst)
        H = Homography._refine(src=dst, dst=dst, initial_H=H)
        return H

    @staticmethod
    def _initialze(*, src: np.ndarray, dst: np.ndarray):
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
    def _refine(
        *, src: np.ndarray, dst: np.ndarray, initial_H: np.ndarray
    ) -> np.ndarray:
        # TODO: See Multiple View Geometry 4.1, 4.2.
        return initial_H


if __name__ == "__main__":
    import os
    import cv2
    from pathlib import Path
    from mvg.calibration import get_chessboard_object_points, find_corners

    np.set_printoptions(suppress=True, precision=5, linewidth=120)

    path = Path(os.environ["DATAPATH"])

    image_points = []
    for root, _, files in os.walk(path):
        rootpath = Path(root)
        for file in files:
            filepath = rootpath / file
            if filepath.suffix == ".jpg":
                image = cv2.imread(str(filepath.absolute()))
                image_points.append(find_corners(image=image, grid_rows=6, grid_cols=9))

    object_points = get_chessboard_object_points(rows=6, cols=9, grid_size=1.0)

    for i, points in enumerate(image_points):
        H1 = Homography.compute(src=object_points[:, :2], dst=points[:, :2])
        H2, _ = cv2.findHomography(object_points[:, :2], points[:, :2])
        dist = np.linalg.norm(H1 - H2)
        print(dist)
        assert dist < 1.0
