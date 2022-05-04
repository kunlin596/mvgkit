"""This modules implements a variety of SfM algorithms
"""
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from mvg import features, streamer
from mvg.camera import Camera
from mvg.mapping.common import frame, reconstruction, visual_odometry


class IncrementalSFM:
    """Incremental Structure from Motion from image stream of monocular camera.

    NOTE: right now it only takes precomputed image feature key points as input.
    TODO: implement backend.
    """

    @dataclass
    class ExecutionState:
        frame_count: int = 0
        fps: float = 0.0  # TODO

    def __init__(
        self,
        *,
        streamer: streamer.StreamerBase,
        camera: Optional[Camera] = None,
        params: Optional[Dict] = None,
    ):
        """
        Args:
            path (Path): path to image key point feature data folder
            camera (camera.Camera): camera model
        """
        self._streamer = streamer
        self._camera = camera

        if self._camera is None:
            self._camera = Camera()

        if params is None:
            params = dict()

        self._reconstruction = None
        self._reconstruction_lock = threading.Lock()
        print("Creating visual odometry object ...")
        self._vo = visual_odometry.VisualOdometry(params.get("visual_odometry", dict()))

        self._state_lock = threading.Lock()
        self._reset_state()

    def _reset_state(self):
        with self._state_lock:
            self._state = self.ExecutionState()

    @property
    def state(self):
        with self._state_lock:
            return self._state

    @property
    def reconstruction(self):
        with self._reconstruction_lock:
            return self._reconstruction

    def _ensure_reconstruction(self):
        with self._reconstruction_lock:
            if self._reconstruction is None:
                print("Creating reconstruction (map) object...")
                self._reconstruction = reconstruction.Reconstruction()

    @property
    def cache(self):
        return dict(visual_odometry=self._vo.cache)

    def step_run(self) -> bool:
        """TODO: virtualize this function."""
        self._ensure_reconstruction()

        frame_data = self._streamer.get()

        if frame_data is None:
            return False

        image = None
        if isinstance(self._streamer, streamer.FeatureFileStreamer):
            keypoints, descriptors = frame_data
        elif isinstance(self._streamer, streamer.ImageFileStreamer):
            keypoints, descriptors = features.SIFT.detect(frame_data.data)
            # FIXME: This is modifying the streamer's buffer.
            # It's ok for now but should be changed in the future.
            frame_data.keypoints = keypoints
            frame_data.descriptors = descriptors
            image = frame_data

        f = frame.Frame(
            id=uuid.uuid4(),
            timestamp=-1.0,
            keypoints=np.asarray(keypoints),
            descriptors=descriptors,
            camera=self._camera,
            image=image,
        )

        succeeded = self._vo.add_frame(reconstruction=self._reconstruction, frame=f)

        print(
            f"Added {self.state.frame_count:>5d}-th frame, id={f.id}, "
            f"current map size: {len(self._reconstruction.landmarks_G):5d}."
        )

        if not succeeded:
            False

        self.state.frame_count += 1
        return True

    def run(self):
        """TODO: virtualize this function."""
        self._ensure_reconstruction()
        self._reset_state()

        print("Start reconstruction ...")
        try:
            i = 0
            while True:
                frame_data = self._streamer.get()

                if frame_data is None:
                    break

                image = None
                if isinstance(self._streamer, streamer.FeatureFileStreamer):
                    keypoints, descriptors = frame_data
                elif isinstance(self._streamer, streamer.ImageFileStreamer):
                    keypoints, descriptors = features.SIFT.detect(frame_data.data)
                    # FIXME: This is modifying the streamer's buffer.
                    # It's ok for now but should be changed in the future.
                    frame_data.keypoints = keypoints
                    frame_data.descriptors = descriptors
                    image = frame_data

                f = frame.Frame(
                    id=uuid.uuid4(),
                    timestamp=-1,
                    keypoints=np.asarray(keypoints),
                    descriptors=descriptors,
                    camera=self._camera,
                    image=image,
                )

                succeeded = self._vo.add_frame(reconstruction=self._reconstruction, frame=f)

                print(
                    f"Added {i:>5d}-th frame, id={f.id}, "
                    f"current map size: {len(self._reconstruction.landmarks_G):5d}."
                )

                if not succeeded:
                    break

                i += 1

        except KeyboardInterrupt:
            print("Keyboard interrupted.")
