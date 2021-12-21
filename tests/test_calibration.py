#!/usr/bin/env python3

import os
import cv2
from pathlib import Path

from mvg.calibration import (
    IntrinsicsCalibration,
    find_corners,
    get_chessboard_object_points,
    compute_reprejection_error,
)
from mvg import camera


def _detect_corners(path: Path):
    all_image_points = []
    for root, _, files in os.walk(path):
        rootpath = Path(root)
        for file in files:
            filepath = rootpath / file
            if filepath.suffix == ".jpg":
                image = cv2.imread(str(filepath.absolute()))
                all_image_points.append(find_corners(image=image, grid_rows=6, grid_cols=9))
    return all_image_points


def _test_calibration_data_set(path: Path, rms_threshold: float):
    all_image_points = _detect_corners(path)
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
        assert abs(reprojection_error) < rms_threshold


def test_rms(data_paths, rms_threshold):
    for path in data_paths:
        _test_calibration_data_set(path, rms_threshold)
