#!/usr/env/bin python3
""" Simple visualizer for Kitti dataset.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv
from mvg.models import kitti


def _main(drive_path, calibration_path, frame_index):
    drive_data = kitti.KittiDrive(Path(drive_path))
    calibration_data = kitti.KittiCalibration(Path(calibration_path))

    images = {
        index: drive_data.read_image(camera_id=index, indices=[frame_index])[0].to_image()
        for index in range(4)
    }

    cameras = {
        camera_id: camera_calibration.get_camera()
        for camera_id, camera_calibration in calibration_data.stereo_calibration.calibrations.items()
    }

    undistorted_images = []
    for camera_id, image in images.items():
        camera = cameras[camera_id]
        undistorted = camera.undistort_image(image, 1.0)
        undistorted_images.append(undistorted)

    plt.figure(0)
    plt.suptitle(f"Camera Images, {Path(drive_path).name}, frame_index={frame_index}")
    for i, image in images.items():
        plt.subplot(2, 2, i + 1)
        plt.title(f"Camera {i}, timestamp={image.timestamp}")
        plt.imshow(image.data)
    plt.tight_layout()

    plt.figure(1)
    plt.suptitle(f"Camera Images, {Path(drive_path).name}, frame_index={frame_index}")
    for i, image in enumerate(undistorted_images):
        plt.subplot(2, 2, i + 1)
        plt.title(f"Camera {i}, timestamp={image.timestamp}")
        plt.imshow(image.data)
    plt.tight_layout()
    plt.show()

    scan = drive_data.read_lidar_scan([frame_index])[0]
    plotter = pv.Plotter()
    plotter.add_points(scan.points)
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument("--drive-path", "-d", help="drive data path", type=str, required=True)
    parser.add_argument(
        "--calibration-path", "-c", help="calibration data path", type=str, required=True
    )
    parser.add_argument("--frame-index", help="frame index", type=int, required=True)

    options = parser.parse_args()

    _main(options.drive_path, options.calibration_path, options.frame_index)
