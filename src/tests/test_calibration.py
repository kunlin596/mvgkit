#!/usr/bin/env python3
import os
import unittest
from pathlib import Path

import numpy as np

from mvgkit.calibration import (
    IntrinsicsCalibration,
    compute_reprejection_error,
    find_corners,
    get_chessboard_object_points,
)
from mvgkit.camera import CameraMatrix
from mvgkit.image_processing import Image
from tests import data_model


class CalibrationTest(unittest.TestCase):
    def left_camera_matrix():
        return CameraMatrix(
            fx=535.35887264164,
            fy=535.6467965086524,
            cx=342.63131029730295,
            cy=233.77551538388735,
            s=0.5369513302298261,
        )

    def right_camera_matrix():
        return CameraMatrix(
            fx=538.7227487607054,
            fy=538.3156703886763,
            cx=327.26717754281043,
            cy=248.63185822830692,
            s=0.10515291690756252,
        )

    def _detect_corners(self, path: Path):
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

    def _test_calibration_data_set(self, path: Path, rms_threshold: float):
        all_image_points, _ = self._detect_corners(path)
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

    def test_intrinsics_calibration_rms(self):
        data_root_path = data_model.DATA_ROOT_PATH
        intrinsics_rms_threshold = data_model.intrinsics_rms_threshold
        data_root_path = Path(data_root_path)
        self._test_calibration_data_set(
            data_root_path / "calibration/left", intrinsics_rms_threshold
        )
        self._test_calibration_data_set(
            data_root_path / "calibration/right", intrinsics_rms_threshold
        )


if __name__ == "__main__":
    np.random.seed(42)
    unittest.main()
