#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mvgkit import features
from mvgkit.algorithms import optical_flow


def _main(path: Path):
    plt.figure(0)

    prev_image = None
    prev_points = None

    for i, filename in enumerate(sorted(os.listdir(path))):
        image_path = path / filename
        curr_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

        if prev_image is None:
            keypoints, _ = features.SIFT.detect(curr_image)
            prev_points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
            prev_image = curr_image
            continue

        start_time = time.time()
        tracked_points, mask = optical_flow.OpticalFlowLK.track(prev_image, curr_image, prev_points)
        elapsed_time = time.time() - start_time
        prev_points = prev_points[mask]
        tracked_points = tracked_points[mask]

        # plt.subplot(211)
        plt.imshow(prev_image)
        plt.scatter(tracked_points[:, 0], tracked_points[:, 1], alpha=0.5, s=2)
        plt.scatter(prev_points[:, 0], prev_points[:, 1], alpha=0.5, s=2)

        for i in range(len(prev_points)):
            end_point = tracked_points[i]
            start_point = prev_points[i]
            d = end_point - start_point
            length = np.linalg.norm(d)
            angle = np.arctan2(d[1], d[0])
            plt.arrow(
                x=start_point[0],
                y=start_point[1],
                dx=np.cos(angle) * length,
                dy=np.sin(angle) * length,
                length_includes_head=True,
                color="r",
                alpha=0.5,
            )
        plt.xlim([0, prev_image.shape[1]])
        plt.ylim([prev_image.shape[0], 0])

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        plt.cla()

        prev_image = curr_image
        prev_points = tracked_points
        print(
            f"Number of tracked points {len(tracked_points):4d}, elapsed time: {elapsed_time:7.3f} sec."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True)

    options = parser.parse_args()

    _main(Path(options.path))
