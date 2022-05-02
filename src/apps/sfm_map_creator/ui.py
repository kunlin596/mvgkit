import os
import threading
import uuid
from pathlib import Path
from typing import Any

from mvg import streamer
from mvg.mapping.mapper.mapper_manager import MapperManager, AvailableMapperType
from mvg.models import kitti
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import MainWindow, QtInteractor


class MapCreatorWindow(MainWindow):
    def __init__(
        self,
        input_path: str,
        calibration_path: str,
        output_path: str,
        parent: Any = None,
        # TODO: use enum.
        streamer_type: str = "image",
        mapper_type: AvailableMapperType = AvailableMapperType.IncrementalSFM,
        show: bool = True,
        interactive: bool = True,
    ):
        QtWidgets.QMainWindow.__init__(self, parent)
        self._init_ui()

        self._input_path = Path(input_path)
        self._calibration_path = Path(calibration_path)
        self._output_path = Path(output_path)

        # TODO: remove hardcoded data type.
        calibration_data = kitti.KittiCalibration(Path(self._calibration_path))
        camera_calibration0 = calibration_data.stereo_calibration.calibrations[0]
        self._output_path = self._output_path / str(uuid.uuid4())
        os.makedirs(self._output_path)

        # Mapper
        self._streamer = self._get_streamer(self._input_path, streamer_type)
        self._mapper = MapperManager.create_mapper(
            mapper_type=mapper_type,
            streamer=self._streamer,
            camera=camera_calibration0.get_unrectified_camera(),
        )
        self._mapper_thread = None
        self._prev_frame_count = 0

        # Monitor
        self._monitor_timer = None
        self._reset_monitor()

        # IPython
        self._ipython_thread = None
        if interactive:
            self._ipython_thread = threading.Thread(target=self._start_ipython)
            self._ipython_thread.start()

        if show:
            self.show()

    def _start_ipython(self):
        # TODO: perhaps manipulate self.
        from IPython import embed

        embed()

    @staticmethod
    def _get_streamer(path: Path, streamer_type: str):
        if streamer_type == "image":
            return streamer.ImageFileStreamer(path)
        elif streamer_type == "feature":
            return streamer.FeatureFileStreamer(path)
        else:
            raise Exception("Not supported streamer type: {streamer_type}!")

    def _reset_monitor(self):
        self._prev_frame_count = 0
        if self._monitor_timer is None:
            self._monitor_timer = QtCore.QTimer()
            self._monitor_timer.timeout.connect(self._update)
        self._monitor_timer.start(50)

    def _run(self):
        if self._mapper_thread is None or not self._mapper_thread.is_alive():
            self._mapper_thread = threading.Thread(target=self._run_mapping)
        else:
            print("Already started.")
            return
        self._mapper_thread.start()

    def __del__(self):
        if self._mapper_thread.is_alive():
            self._mapper_thread.join()
        if self._ipython_thread is not None and self._ipython_thread.is_alive():
            self._ipython_thread.join()

    def _run_mapping(self):
        self._mapper.run()

    def _update(self):
        """
        TODO: implement visualization module.
        """
        if self._mapper.state.frame_count != self._prev_frame_count:
            self._plotter.clear()
            points = self._mapper.reconstruction.get_landmark_positions_G()
            if len(points):
                self._plotter.add_points(
                    points, point_size=5.0, color="g", render_points_as_spheres=True
                )

    def _step_run(self):
        self._mapper.step_run()

    def _step_next(self):
        self._mapper.step_run()

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
        run_button.setShortcut("Ctrl+Shift+R")
        run_button.triggered.connect(self._run)
        run_menu.addAction(run_button)

        step_button = QtGui.QAction("Step Run", self)
        step_button.setShortcut("Ctrl+R")
        step_button.triggered.connect(self._step_run)
        run_menu.addAction(step_button)

        step_next_button = QtGui.QAction("Step Next", self)
        step_next_button.setShortcut("Ctrl+N")
        step_next_button.triggered.connect(self._step_next)
        run_menu.addAction(step_next_button)
