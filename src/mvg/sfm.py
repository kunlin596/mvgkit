"""This modules implements a variety of SfM algorithms
"""
import uuid
from pathlib import Path

import numpy as np

from mvg import camera, features, streamer
from mvg.mapping import frame, reconstruction, visual_odometry
import threading


class IncrementalSFM:
    """Incremental Structure from Motion from image stream of monocular camera.

    NOTE: right now it only takes precomputed image feature key points as input.
    TODO: implement backend.
    """

    def __init__(self, streamer: streamer.StreamerBase, camera: camera.Camera, output_path: Path):
        """
        Args:
            path (Path): path to image key point feature data folder
            camera (camera.Camera): camera model
        """
        self._streamer = streamer
        self._camera = camera
        self._output_path = output_path
        self._reconstruction_lock = threading.Lock()

    @property
    def reconstruction(self):
        with self._reconstruction_lock:
            return self._reconstruction

    def run(self):
        print("Creating reconstruction and visual odometry...")
        self._reconstruction = reconstruction.Reconstruction()
        self._vo = visual_odometry.VisualOdometry()
        print("Start reconstruction ...")

        try:
            i = 0
            while True:
                new_data = self._streamer.get()

                if new_data is None:
                    break

                if isinstance(self._streamer, streamer.FeatureFileStreamer):
                    keypoints, descriptors = new_data
                elif isinstance(self._streamer, streamer.ImageFileStreamer):
                    keypoints, descriptors = features.SIFT.detect(new_data)

                f = frame.Frame(
                    id=uuid.uuid4(),
                    timestamp=-1,
                    keypoints=np.asarray(keypoints),
                    descriptors=descriptors,
                    camera=self._camera,
                )

                print(
                    f"Adding {i:>5d}-th frame, id={f.id} "
                    f"current map size: {len(self._reconstruction.landmarks):10d}."
                )
                succeeded = self._vo.add_frame(reconstruction=self._reconstruction, frame=f)

                if not succeeded:
                    break

                i += 1

        except KeyboardInterrupt:
            print("Keyboard interrupted.")
        finally:
            # self._reconstruction.dump(self._output_path)
            pass
