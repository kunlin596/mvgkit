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
from mvg.camera import CameraMatrix
from pytest import fixture


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
    print(camera_matrix)

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


@fixture()
def left_camera_matrix():
    return CameraMatrix(
        fx=535.35887264164,
        fy=535.6467965086524,
        cx=342.63131029730295,
        cy=233.77551538388735,
        s=0.5369513302298261,
    )


@fixture()
def right_camera_matrix():
    return CameraMatrix(
        fx=538.7227487607054,
        fy=538.3156703886763,
        cx=327.26717754281043,
        cy=248.63185822830692,
        s=0.10515291690756252,
    )
