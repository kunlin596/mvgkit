#!/usr/env/bin python3
""" Simple visualizer for Kitti dataset.
"""
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from mvgkit.features import SIFT, Matcher
from mvgkit.models import kitti
from mvgkit.stereo import Fundamental

np.set_printoptions(suppress=True, precision=7, linewidth=120)


def _read_data(drive_path, calibration_path, frame_index):
    drive_data = kitti.KittiDrive(Path(drive_path))
    calibration_data = kitti.KittiCalibration(Path(calibration_path))

    images = {
        index: drive_data.read_image(camera_id=index, indices=[frame_index])[0].to_image()
        for index in range(4)
    }

    camera_calibrations = {
        camera_id: camera_calibration
        for camera_id, camera_calibration in calibration_data.stereo_calibration.calibrations.items()
    }

    lidar_scan = drive_data.read_lidar_scan([frame_index])[0]
    lidar_calibration = calibration_data.lidar_calibration

    return images, camera_calibrations, lidar_scan, lidar_calibration


def _plot_epilines_image_pair(images, indices, sample_step):
    image0 = images[indices[0]]
    image1 = images[indices[1]]

    features0, descriptors0 = SIFT.detect(image0.data)
    features1, descriptors1 = SIFT.detect(image1.data)

    matches = Matcher.match(descriptors0, descriptors1)
    image_points_L, image_points_R, _ = Matcher.get_matched_points(features0, features1, matches)

    F_RL, inlier_mask = Fundamental.compute(x_L=image_points_L, x_R=image_points_R)

    Fundamental.plot_epipolar_lines(
        image_L=image0.data,
        image_R=image1.data,
        points_L=image_points_L[inlier_mask][::sample_step],
        points_R=image_points_R[inlier_mask][::sample_step],
        F_RL=F_RL,
    )


def _plot_epilines(images, sample_step=5):
    _plot_epilines_image_pair(images, [0, 1], sample_step)
    _plot_epilines_image_pair(images, [2, 3], sample_step)


def _project_points_to_image(points, P):
    points = points[points[:, 2] > 0.0]
    image_points = points[:, :2] / points[:, -1].reshape(-1, 1)
    image_points = image_points @ P[:2, :2].T + P[:2, 2]
    return image_points


def _plot_overlay(
    images,
    camera_calibrations: List[kitti.KittiCameraCalibration],
    lidar_scan,
    lidar_calibration,
):
    lidar_pose_C0 = lidar_calibration.T
    camera_calibration0 = camera_calibrations[0]
    points_C0 = camera_calibration0.R_rectification.apply(lidar_pose_C0 @ lidar_scan.points)

    lidar_pose_C2 = camera_calibrations[2].T.inv() @ lidar_calibration.T
    camera_calibration2 = camera_calibrations[2]
    points_C2 = camera_calibration2.R_rectification.apply(lidar_pose_C2 @ lidar_scan.points)

    image_points_C0 = _project_points_to_image(points_C0, camera_calibration0.P)
    image_points_C2 = _project_points_to_image(points_C2, camera_calibration2.P)

    image0 = images[0].data
    image2 = images[2].data

    plt.figure(0)

    plt.subplot(211)
    plt.imshow(image0)
    plt.scatter(image_points_C0[:, 0], image_points_C0[:, 1], s=0.3, alpha=0.5)
    plt.xlim([0, image0.shape[1]])
    plt.ylim([image0.shape[0], 0])

    plt.subplot(212)
    plt.imshow(image2)
    plt.scatter(image_points_C2[:, 0], image_points_C2[:, 1], s=0.3, alpha=0.5)
    plt.xlim([0, image2.shape[1]])
    plt.ylim([image2.shape[0], 0])

    plt.show()


def _main(drive_path, calibration_path, frame_index, epilines, overlay):
    images, camera_calibrations, lidar_scan, lidar_calibration = _read_data(
        drive_path, calibration_path, frame_index
    )

    if epilines:
        _plot_epilines(images)

    elif overlay:
        _plot_overlay(images, camera_calibrations, lidar_scan, lidar_calibration)

    # plt.figure(0)
    # plt.suptitle(f"Camera Images, {Path(drive_path).name}, frame_index={frame_index}")
    # for i, image in images.items():
    #     plt.subplot(2, 2, i + 1)
    #     plt.title(f"Camera {i}, timestamp={image.timestamp}")
    #     plt.imshow(image.data)
    # plt.tight_layout()
    # plt.show()

    plotter = pv.Plotter()
    plotter.add_points(lidar_scan.points)
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument("--drive-path", "-d", help="drive data path", type=str, required=True)
    parser.add_argument(
        "--calibration-path", "-c", help="calibration data path", type=str, required=True
    )
    parser.add_argument("--frame-index", help="frame index", type=int, required=True)
    parser.add_argument("--epilines", "-l", action="store_true")
    parser.add_argument("--overlay", "-o", action="store_true")

    options = parser.parse_args()

    _main(
        options.drive_path,
        options.calibration_path,
        options.frame_index,
        options.epilines,
        options.overlay,
    )
