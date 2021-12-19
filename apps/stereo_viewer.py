#!/usr/bin/env python3

from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtUiTools import QUiLoader
from PySide2.QtMultimedia import QCamera
from pathlib import Path

import argparse
import time
import sys
import cv2
import pickle
from typing import List, Optional

from mvg.calibration import CvIntrinsicsCalibrationData, undistort_image


class StereoViewer(QtCore.QObject):
    _is_recording = False

    def __init__(
        self,
        *,
        intrinsics_calibration_files: Optional[List[str]] = None,
        stereo_calibration_file: Optional[str] = None,
    ):
        QtCore.QObject.__init__(self)
        self._cameras = dict()
        self._cached_frames = dict()
        self.setup_ui()
        self._load_intrinsics_calibration_files(filenames=intrinsics_calibration_files)
        self._load_stereo_calibration_file(filename=stereo_calibration_file)

    def _load_intrinsics_calibration_files(
        self, *, filenames: Optional[List[str]] = None
    ):
        self._calibration_data_0 = None
        self._calibration_data_1 = None
        if filenames is None:
            return

        with open(filenames[0], "rb") as f:
            self._calibration_data_0 = pickle.load(f)

        with open(filenames[1], "rb") as f:
            self._calibration_data_1 = pickle.load(f)

        if (
            self._calibration_data_0 is not None
            and self._calibration_data_1 is not None
        ):
            self._undistort_checkbox.setEnabled(True)

    def _load_stereo_calibration_file(self, *, filename: Optional[str] = None):
        self._stereo_calibration_data = None
        if filename is None:
            return

        with open(Path(filename).absolute(), "rb") as f:
            self._stereo_calibration_data = pickle.load(f)

        if self._stereo_calibration_data is not None:
            self._rectification_checkbox.setEnabled(True)

    def setup_ui(self):
        loader = QUiLoader()
        ui_file_path = Path(__file__).parent.absolute() / "stereo_viewer.ui"
        assert ui_file_path.exists()
        ui_file = QtCore.QFile(str(ui_file_path))
        ui_file.open(QtCore.QFile.ReadOnly)
        self._window = loader.load(ui_file)
        ui_file.close()

        self._camera0_list = self._window.findChild(
            QtWidgets.QComboBox, "camera0_combobox"
        )
        assert self._camera0_list is not None
        self._camera1_list = self._window.findChild(
            QtWidgets.QComboBox, "camera1_combobox"
        )
        assert self._camera1_list is not None
        self._image0 = self._window.findChild(QtWidgets.QLabel, "camera0_image")
        assert self._image0 is not None
        self._image1 = self._window.findChild(QtWidgets.QLabel, "camera1_image")
        assert self._image1 is not None

        # print(QCameraInfo.availableDevices())
        for camerainfo in QCamera.availableDevices():
            self._cameras[str(camerainfo, encoding="ascii")] = QCamera(camerainfo)

        for key, _ in self._cameras.items():
            self._camera0_list.addItem(key)
            self._camera1_list.addItem(key)

        self._camera0_list.currentTextChanged.connect(self._set_camera0)
        self._camera1_list.currentTextChanged.connect(self._set_camera1)

        self._start_record_button = self._window.findChild(
            QtWidgets.QPushButton, "start_record_button"
        )
        self._start_record_button.clicked.connect(self._start_record)

        self._stop_record_button = self._window.findChild(
            QtWidgets.QPushButton, "stop_record_button"
        )
        self._stop_record_button.clicked.connect(self._stop_record)

        self._save_record_button = self._window.findChild(
            QtWidgets.QPushButton, "save_record_button"
        )
        self._save_record_button.clicked.connect(self._save_record)

        self._start_record_button.setEnabled(True)
        self._stop_record_button.setEnabled(False)
        self._save_record_button.setEnabled(False)

        self._capture_timer = QtCore.QTimer()

        self._undistort_checkbox = self._window.findChild(
            QtWidgets.QCheckBox, "undistort_checkbox"
        )

        self._rectification_checkbox = self._window.findChild(
            QtWidgets.QCheckBox, "rectification_checkbox"
        )

    def show(self):
        self._window.show()

    def _capture(self, index):
        _, frame = getattr(self, f"_camera{index}").read()
        if frame is None or frame.size == 0:
            return

        if self._is_recording:
            self._cached_frames[index].append(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        if self._undistort_checkbox.isChecked():
            frame = undistort_image(getattr(self, f"_calibration_data_{index}"), frame)
            # FIXME: ValueError: ndarray is not C-contiguous
            frame = frame.copy(order="C")

        if self._rectification_checkbox.isChecked():
            # TODO
            pass

        image = QtGui.QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        label = getattr(self, f"_image{index}")
        pix_map = QtGui.QPixmap.fromImage(image)
        scaled = pix_map.scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def _capture0(self):
        self._capture(0)

    def _capture1(self):
        self._capture(1)

    def _set_camera(self, index, text):
        setattr(self, f"_camera{index}", cv2.VideoCapture(int(text[-1])))
        self._capture_timer.timeout.connect(getattr(self, f"_capture{index}"))
        self._capture_timer.start(30)

    def _set_camera0(self, text):
        self._set_camera(0, text)

    def _set_camera1(self, text):
        self._set_camera(1, text)

    def _start_record(self):
        self._is_recording = True
        self._cached_frames = {0: [], 1: []}
        self._start_record_button.setEnabled(False)
        self._stop_record_button.setEnabled(True)
        self._save_record_button.setEnabled(False)
        print("Start recording...")

    def _stop_record(self):
        self._is_recording = False
        self._start_record_button.setEnabled(False)
        self._stop_record_button.setEnabled(False)
        self._save_record_button.setEnabled(True)
        print("Stopped recording.")

    def _save_record(self):
        filename = f"/tmp/stereo_pack_{time.time():7.3f}.data"
        with open(filename, "bw") as f:
            if len(self._cached_frames):
                pickle.dump(self._cached_frames, f)
        self._start_record_button.setEnabled(True)
        self._stop_record_button.setEnabled(False)
        self._save_record_button.setEnabled(False)
        print(f"Saved to [{filename}].")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intrinsics-calibration-files",
        "-i",
        nargs="+",
        type=str,
        help="Intrinsics calibration data file for left and right eyes.",
    )
    parser.add_argument(
        "--stereo-calibration-file",
        "-s",
        type=str,
        help="Stereo calibration file for stereo camera pair.",
    )
    options = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    viewer = StereoViewer(
        intrinsics_calibration_files=options.intrinsics_calibration_files,
        stereo_calibration_file=options.stereo_calibration_file,
    )
    viewer.show()
    sys.exit(app.exec_())
