#!/usr/bin/env python3
import os
from pathlib import Path

from mvgkit.calibration import (
    IntrinsicsCalibration,
    compute_reprejection_error,
    find_corners,
    get_chessboard_object_points,
)
from mvgkit.image_processing import Image


def _detect_corners(path: Path):
    all_image_points = []
    all_images = []
    for root, _, files in os.walk(path):
        rootpath = Path(root)
        for file in sorted(files):
            filepath = rootpath / file
            if filepath.suffix == ".jpg":
                image = Image.from_file(str(filepath.absolute())).data
                all_images.append(image)
                all_image_points.append(find_corners(image=image, grid_rows=6, grid_cols=9))
    return all_image_points, all_images


def _test_calibration_data_set(path: Path, rms_threshold: float):
    all_image_points, _ = _detect_corners(path)
    object_points = get_chessboard_object_points(rows=6, cols=9, grid_size=1.0)
    camera_matrix, radial_distortion_model, debuginfo = IntrinsicsCalibration.calibrate(
        all_image_points, object_points, debug=True
    )
    print(camera_matrix)
    all_extrinsics = debuginfo["all_extrinsics"]
    for i, image_points in enumerate(all_image_points):
        T_CW = all_extrinsics[i]
        reprojection_error = compute_reprejection_error(
            image_points=image_points,
            object_points_W=object_points,
            camera_matrix=camera_matrix,
            T_CW=T_CW,
            radial_distortion_model=radial_distortion_model,
        )
        print(f"reprojection_error={reprojection_error:6.3f}")
        assert reprojection_error < rms_threshold, f"rms_threshold={rms_threshold:7.3f}."


def test_intrinsics_calibration_rms(data_root_path, intrinsics_rms_threshold):
    _test_calibration_data_set(data_root_path / "calibration/left", intrinsics_rms_threshold)
    _test_calibration_data_set(data_root_path / "calibration/right", intrinsics_rms_threshold)
