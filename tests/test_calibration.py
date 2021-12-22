#!/usr/bin/env python3

import os
import cv2
from pathlib import Path
from mvg.basic import homogeneous

from mvg.calibration import (
    IntrinsicsCalibration,
    find_corners,
    get_chessboard_object_points,
    compute_reprejection_error,
)
from mvg.fundamental import Fundamental
from mvg import camera


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


def test_intrinsics_calibration_rms(data_paths, intrinsics_rms_threshold):
    for path in data_paths:
        _test_calibration_data_set(path, intrinsics_rms_threshold)


def test_fundamental_matrix(data_paths, fundamental_rms_threshold):
    all_image_points_l, left_images = _detect_corners(data_paths[0])
    all_image_points_r, right_images = _detect_corners(data_paths[1])

    print("\nComputing OpenCV F for performance reference.")
    for i, (image_points_l, image_points_r) in enumerate(
        zip(all_image_points_l, all_image_points_r)
    ):
        F = Fundamental.compute(x=image_points_l, x2=image_points_r)
        rms = Fundamental.compute_rms(F=F, x=image_points_l, x2=image_points_r)
        F2 = cv2.findFundamentalMat(image_points_l, image_points_r)[0]
        rms2 = Fundamental.compute_rms(F=F2, x=image_points_l, x2=image_points_r)

        print(f"rms={rms:7.3f}, opencv_rms={rms2:7.3f}, {'Won' if rms < rms2 else 'Lost'}")
        assert rms < fundamental_rms_threshold  # in pixel
