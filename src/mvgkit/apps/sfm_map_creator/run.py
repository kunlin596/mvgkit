#!/usr/bin/env python3
"""
"""
import argparse
import sys

import numpy as np
from apps.sfm_map_creator.ui import MapCreatorWindow
from PySide6 import QtWidgets

if __name__ == "__main__":

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Input data path.")
    parser.add_argument("--output-path", "-o", type=str, required=True, help="Output result path.")
    parser.add_argument(
        "--calibration-path", "-c", type=str, required=True, help="Input calibration data path."
    )
    parser.add_argument(
        "--streamer-type",
        "-s",
        type=str,
        help="Data streamer type.",
        choices=["image", "feature"],
        default="image",
    )
    parser.add_argument("--noui", "-n", action="store_true", help="No UI mode or not.")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode or not.")
    parser.add_argument("--debug", action="store_true", help="Debug mode or not.")

    options = parser.parse_args()

    if options.debug:
        np.set_printoptions(suppress=True, precision=5)

    app = QtWidgets.QApplication([])
    window = MapCreatorWindow(
        input_path=options.input_path,
        calibration_path=options.calibration_path,
        output_path=options.output_path,
        streamer_type=options.streamer_type,
        show=not options.noui,
        interactive=options.interactive,
        debug=options.debug,
    )
    sys.exit(app.exec_())
