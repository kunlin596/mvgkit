#!/usr/bin/env python3


import argparse
import pickle
from pathlib import Path

from mvg import calibration


def read(filename: Path):
    assert filename.exists()
    with open(filename, "rb") as f:
        return pickle.load(f)


def main(*, options):
    filename = Path(options.filename)
    data = read(filename)
    assert data is not None
    # FIXME: calibrateCamera's performance drops dramatically when number of images increases,
    # hard code a step size for now.
    for i in range(2):
        print(f"Calibrating intrinsics of camera {i:d}")
        calibration_data = calibration.intrinsic_calibration(images=data[i][::10])
        with open(
            filename.parent.absolute() / f"{str(filename):s}.{i:d}.calib", "wb"
        ) as f:
            pickle.dump(calibration_data, f)
    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--filename",
        "-f",
        required=True,
        type=str,
        help="Path to data for calibration.",
    )
    parser.add_argument(
        "--frame-number", "-n", default=-1, type=int, help="Calibrate specified frame"
    )

    options = parser.parse_args()
    main(options=options)
