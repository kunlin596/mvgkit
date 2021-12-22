#!/usr/bin/env python3

import os
import cv2
import numpy as np
from pathlib import Path
from mvg import features

from mvg.calibration import (
    IntrinsicsCalibration,
    find_corners,
    get_chessboard_object_points,
    compute_reprejection_error,
)
from mvg.fundamental import Fundamental


def _detect_corners(path: Path):
    all_image_points = []
    all_images = []
    for root, _, files in os.walk(path):
        rootpath = Path(root)
        for file in sorted(files):
            filepath = rootpath / file
            if filepath.suffix == ".jpg":
                image = cv2.imread(str(filepath.absolute()))
                all_images.append(image)
                all_image_points.append(find_corners(image=image, grid_rows=6, grid_cols=9))
    return all_image_points, all_images


def _test_calibration_data_set(path: Path, rms_threshold: float):
    all_image_points, _ = _detect_corners(path)
    object_points = get_chessboard_object_points(rows=6, cols=9, grid_size=1.0)
    camera_matrix, radial_distortion_model, debuginfo = IntrinsicsCalibration.calibrate(
        all_image_points, object_points, debug=True
    )

    all_extrinsics = debuginfo["all_extrinsics"]
    for i, image_points in enumerate(all_image_points):
        pose = all_extrinsics[i]
        reprojection_error = compute_reprejection_error(
            image_points=image_points,
            object_points=object_points,
            camera_matrix=camera_matrix,
            camera_pose=pose,
            radial_distortion_model=radial_distortion_model,
        )
        print(f"reprojection_error={reprojection_error:6.3f}")
        assert reprojection_error < rms_threshold


def test_intrinsics_calibration_rms(data_root_path, intrinsics_rms_threshold):
    data_root_path = Path(data_root_path)
    _test_calibration_data_set(data_root_path / "calibration/left", intrinsics_rms_threshold)
    _test_calibration_data_set(data_root_path / "calibration/right", intrinsics_rms_threshold)


def test_fundamental_matrix(data_root_path, fundamental_rms_threshold):
    print("Computing feature points on left and right images...")
    caliration_root_path = Path(data_root_path) / "calibration"
    caliration_left_path = caliration_root_path / "left"
    caliration_right_path = caliration_root_path / "right"

    all_image_points_l = []
    all_image_points_r = []
    all_images_left = []
    all_images_right = []
    for file1, file2 in zip(
        sorted(os.listdir(caliration_left_path)), sorted(os.listdir(caliration_right_path))
    ):
        image_left = cv2.imread(str(caliration_left_path / file1))
        image_right = cv2.imread(str(caliration_right_path / file2))
        all_images_left.append(image_left)
        all_images_right.append(image_right)
        keypoints1, descriptors1 = features.SIFT.detect(image_left)
        keypoints2, descriptors2 = features.SIFT.detect(image_right)
        matches = features.Matcher.match(descriptors1=descriptors1, descriptors2=descriptors2)
        points1 = []
        points2 = []
        good_matches = []
        for m, n in matches:
            if m.distance < 0.35 * n.distance:
                good_matches.append(m)
                points1.append(keypoints1[m.queryIdx].pt)
                points2.append(keypoints2[m.trainIdx].pt)
        assert len(points1) > 8 and len(points2) > 8 and len(points1) == len(points2)
        all_image_points_l.append(np.asarray(points1))
        all_image_points_r.append(np.asarray(points2))

    all_image_points_l = all_image_points_l
    all_image_points_r = all_image_points_r

    print("\nComputing fundamental matrix and use OpenCV F for performance reference.")
    for i, (image_points_l, image_points_r) in enumerate(
        zip(all_image_points_l, all_image_points_r)
    ):
        F = Fundamental.compute(x=image_points_l, x2=image_points_r)
        rms = Fundamental.compute_rms(F=F, x=image_points_l, x2=image_points_r)
        F2 = cv2.findFundamentalMat(image_points_l, image_points_r)[0]
        rms2 = Fundamental.compute_rms(F=F2, x=image_points_l, x2=image_points_r)

        samples = list(set(np.random.randint(0, len(image_points_l), 10)))
        Fundamental.plot_epipolar_lines(
            all_images_left[i],
            all_images_right[i],
            image_points_l[samples],
            image_points_r[samples],
            F,
        )

        print(f"rms={rms:7.3f}, opencv_rms={rms2:7.3f}, {'Won' if rms < rms2 else 'Lost'}")
        assert rms < fundamental_rms_threshold  # in pixel
