import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mvg import streamer
from mvg.basic import SE3
from mvg.mapping.mapper.mapper_manager import AvailableMapperType, MapperManager
from mvg.models import kitti
from PySide6 import QtCore, QtGui, QtWidgets
from pyvistaqt import MainWindow, QtInteractor


def _plot_pose(plotter, pose: SE3):
    for i in range(3):
        plotter.add_arrows(
            cent=pose.t,
            direction=pose.R.as_matrix().T[i],
            mag=1,
            color=["r", "g", "b"][i],
            render_lines_as_tubes=True,
        )


class _ExecutionMode(IntEnum):
    Consecutive = 0
    Step = 1


@dataclass
class _ExecutionState:
    mode: _ExecutionMode = _ExecutionMode.Consecutive


class MapCreatorWindow(MainWindow):

    _image_update_signal = QtCore.Signal()

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
        debug: bool = True,
    ):
        QtWidgets.QMainWindow.__init__(self, parent)
        try:
            self._init_ui()
        except AttributeError as e:
            print(f"UI setup failed! e={e}.")
            raise

        self._image_update_signal.connect(self._show_image)

        self._state = _ExecutionState()

        self._input_path = Path(input_path)
        self._calibration_path = Path(calibration_path)
        self._output_path = Path(output_path)

        # TODO: remove hardcoded data type.
        calibration_data = kitti.KittiCalibration(Path(self._calibration_path))
        camera_calibration0 = calibration_data.stereo_calibration.calibrations[0]
        self._output_path = self._output_path / str(uuid.uuid4())
        os.makedirs(self._output_path)

        # Mapper
        self._debug = debug
        self._streamer = self._get_streamer(self._input_path, streamer_type)
        self._mapper = MapperManager.create_mapper(
            mapper_type=mapper_type,
            streamer=self._streamer,
            camera=camera_calibration0.get_unrectified_camera(),
            params=dict(
                visual_odometry=dict(
                    debug=self._debug,
                ),
            ),
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
        self._state.mode = _ExecutionMode.Consecutive
        while True:
            if not self._step_run():
                break
            time.sleep(0.01)

    def _update(self):
        """
        TODO: implement visualization module.
        """
        if self._mapper.state.frame_count != self._prev_frame_count:
            self._prev_frame_count = self._mapper.state.frame_count

            if self._debug:
                # NOTE: The points are estimated in reference frame.
                frames = self._mapper.reconstruction.frames
                if len(frames) < 2:
                    return
                ref_frame = frames[-2]
                points3d = ref_frame.points3d
                if points3d is None or len(points3d) == 0:
                    return

                pose_G = ref_frame.pose_G
                image_points = ref_frame.camera.project_points(points3d)
                image_points = image_points.round().astype(np.int32)

                colors = (
                    ref_frame.image.data[image_points[:, 1], image_points[:, 0]]
                    .reshape(-1, 3)
                    .astype(np.float)
                )

                self._plotter.add_points(
                    pose_G @ points3d,
                    # render_points_as_spheres=True,
                    scalars=colors,
                    rgb=True,
                    point_size=5.0,
                )

                _plot_pose(self._plotter, pose_G)

            else:
                points = self._mapper.reconstruction.get_landmark_positions_G()
                if len(points):
                    self._plotter.add_points(
                        points, point_size=5.0, color="g", render_points_as_spheres=True
                    )

    def _step_run(self):
        self._state.mode = _ExecutionMode.Step
        r = self._mapper.step_run()
        self._image_update_signal.emit()
        return r

    def _show_image(self):
        if self._view_image_dialog is None:
            self._view_image_dialog = QtWidgets.QDialog()
            view_image_main = QtWidgets.QWidget()
            # FIXME: should the canvas be cached?
            self._image_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            vboxlayout = QtWidgets.QVBoxLayout(view_image_main)
            vboxlayout.setContentsMargins(0, 0, 0, 0)
            vboxlayout.addWidget(NavigationToolbar(self._image_canvas, self))
            vboxlayout.addWidget(self._image_canvas)
            self._view_image_dialog.setLayout(vboxlayout)

        fig = self._image_canvas.figure
        fig.clear()
        axes = fig.subplots(2, 1)
        if len(self._streamer.frame_buffer) > 0:
            ax = axes[0]
            frame = self._streamer.frame_buffer[-1]
            ax.set_title(Path(frame.uri).name)
            ax.imshow(frame.data, cmap="gray")
            if frame.keypoints is not None and len(frame.keypoints):
                keypoints = np.asarray([kp.pt for kp in frame.keypoints])
                ax.scatter(keypoints[:, 0], keypoints[:, 1], s=5.0, alpha=0.5)

        if len(self._streamer.frame_buffer) > 1:
            ax = axes[1]
            frame = self._streamer.frame_buffer[-2]
            ax.set_title(Path(frame.uri).name)
            ax.imshow(frame.data, cmap="gray")
            keypoints = np.asarray([kp.pt for kp in frame.keypoints])
            ax.scatter(keypoints[:, 0], keypoints[:, 1], s=5.0, alpha=0.5)

        fig.tight_layout()
        self._image_canvas.draw()
        self._view_image_dialog.show()

    def _init_ui(self):
        self._main_frame = QtWidgets.QFrame()
        vboxlayout = QtWidgets.QVBoxLayout()
        vboxlayout.setContentsMargins(0, 0, 0, 0)

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
        run_button.setShortcut("Ctrl+C")
        run_button.triggered.connect(self._run)
        run_menu.addAction(run_button)

        step_button = QtGui.QAction("Run", self)
        step_button.setShortcut("Ctrl+R")
        step_button.triggered.connect(self._step_run)
        run_menu.addAction(step_button)

        view_menu = main_menu.addMenu("View")
        view_image_button = QtGui.QAction("Show Image", self)
        view_image_button.setShortcut("Ctrl+V")
        view_image_button.triggered.connect(self._show_image)
        view_menu.addAction(view_image_button)

        self._view_image_dialog = None
