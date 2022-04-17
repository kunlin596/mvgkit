#!/usr/env/bin python3
""" Simple visualizer for Kitti dataset.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mvg.models import kitti


def _main(drive_path, calibration_path, frame_index):
    drive_data = kitti.KittiDrive(Path(drive_path))
    # calibration_data = kitti.KittiCalibration(Path(calibration_path))

    images = [
        drive_data.read_image(camera_id=index, indices=[frame_index])[0] for index in range(4)
    ]

    plt.figure(0)
    plt.suptitle(f"Camera Images, {Path(drive_path).name}, frame_index={frame_index}")
    for i, image in enumerate(images):
        plt.subplot(2, 2, i + 1)
        plt.title(f"Camera {i}, timestamp={image.timestamp}")
        plt.imshow(image.data)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument("--drive-path", "-d", help="drive data path", type=str, required=True)
    parser.add_argument(
        "--calibration-path", "-c", help="calibration data path", type=str, required=True
    )
    parser.add_argument("--frame-index", help="frame index", type=int, required=True)

    options = parser.parse_args()

    _main(options.drive_path, options.calibration_path, options.frame_index)
