#!/usr/bin/env python3

import argparse
import os
import uuid
from pathlib import Path

from mvg.mapping.mapper import sfm
from mvg.models import kitti
from mvg.streamer import ImageFileStreamer


def _main(input_dir, calibration_path, output_dir):
    calibration_data = kitti.KittiCalibration(Path(calibration_path))
    camera_calibration0 = calibration_data.stereo_calibration.calibrations[0]
    output_path = output_dir / str(uuid.uuid4())
    os.makedirs(output_path)
    mapper = sfm.IncrementalSFM(
        streamer=ImageFileStreamer(input_dir),
        camera=camera_calibration0.get_unrectified_camera(),
    )
    mapper.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--calibration-path", "-c", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)

    options = parser.parse_args()

    _main(Path(options.input), Path(options.calibration_path), Path(options.output))
