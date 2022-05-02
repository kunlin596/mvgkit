import os
import threading
import uuid
from pathlib import Path
from typing import Any

from mvg import sfm, streamer
from mvg.models import kitti
from PySide6 import QtGui, QtWidgets, QtCore
from pyvistaqt import MainWindow, QtInteractor


class MapCreatorWindow(MainWindow):
    def __init__(
        self,
        input_path: str,
        calibration_path: str,
        output_path: str,
        parent: Any = None,
        streamer_type: str = "image",
        show: bool = True,
        # TODO: use enum.
    ):
        QtWidgets.QMainWindow.__init__(self, parent)
        self._init_ui()

        self._input_path = Path(input_path)
        self._calibration_path = Path(calibration_path)
        self._output_path = Path(output_path)

        self._streamer = self._get_streamer(self._input_path, streamer_type)

        calibration_data = kitti.KittiCalibration(Path(self._calibration_path))
        camera_calibration0 = calibration_data.stereo_calibration.calibrations[0]

        self._output_path = self._output_path / str(uuid.uuid4())

        os.makedirs(self._output_path)
        self._mapper = sfm.IncrementalSFM(
            self._streamer, camera_calibration0.get_unrectified_camera(), output_path
        )

        self._mapper_thread = None
        self._monitor_timer = None
        self._prev_frame_count = 0

        if show:
            self.show()

    @staticmethod
    def _get_streamer(path: Path, streamer_type: str):
        if streamer_type == "image":
            return streamer.ImageFileStreamer(path)
        elif streamer_type == "feature":
            return streamer.FeatureFileStreamer(path)
        else:
            raise Exception("Not supported streamer type: {streamer_type}!")

    def _start(self):
        if self._mapper_thread is None or not self._mapper_thread.is_alive():
            self._mapper_thread = threading.Thread(target=self._run_mapping)
        else:
            print("Already started.")
            return
        self._mapper_thread.start()

        self._prev_frame_count = 0
        if self._monitor_timer is None:
            self._monitor_timer = QtCore.QTimer()
            self._monitor_timer.timeout.connect(self._update)
        self._monitor_timer.start(50)

    def __del__(self):
        if self._mapper_thread.is_alive():
            self._mapper_thread.join()

    def _run_mapping(self):
        self._mapper.run()

    def _update(self):
        if len(self._mapper.reconstruction.frames) != self._prev_frame_count:
            self._plotter.clear()
            points = self._mapper.reconstruction.get_landmark_positions_G()
            if len(points):
                self._plotter.add_points(
                    points, point_size=5.0, color="g", render_points_as_spheres=True
                )

    def _init_ui(self):
        self._main_frame = QtWidgets.QFrame()
        vboxlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self._plotter = QtInteractor(self._main_frame)
        self._plotter.set_position([0, 0, 0])
        self._plotter.set_focus([0, 0, 1])
        self._plotter.set_viewup([0, -1, 0])
        self._plotter.add_axes_at_origin()
        self._plotter.add_axes()

        vboxlayout.addWidget(self._plotter.interactor)
        self.signal_close.connect(self._plotter.close)

        self._main_frame.setLayout(vboxlayout)
        self.setCentralWidget(self._main_frame)

        main_menu = self.menuBar()

        # File menu
        file_menu = main_menu.addMenu("File")
        exit_button = QtGui.QAction("Exit", self)
        exit_button.setShortcut("Ctrl+Q")
        exit_button.triggered.connect(self.close)
        file_menu.addAction(exit_button)

        # Execution
        run_menu = main_menu.addMenu("Run")
        run_button = QtGui.QAction("Run", self)
        run_button.setShortcut("Ctrl+R")
        run_button.triggered.connect(self._start)
        run_menu.addAction(run_button)
