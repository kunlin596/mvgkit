"""This modules implements a variety of SfM algorithms
"""
import uuid
from pathlib import Path

import numpy as np

from mvg import camera, features
from mvg.mapping import frame, reconstruction, visual_odometry


class FeatureTracker:
    pass


class IncrementalSFM:
    """Incremental Structure from Motion from image stream of monocular camera.

    NOTE: right now it only takes precomputed image feature key points as input.
    TODO: implement backend.
    """

    def __init__(self, input_path: Path, camera: camera.Camera, output_path: Path):
        """
        Args:
            path (Path): path to image key point feature data folder
            camera (camera.Camera): camera model
        """
        self._frame_reader = features.get_feature_iter(input_path)
        self._camera = camera
        self._output_path = output_path

    def run(self):
        self._reconstruction = reconstruction.Reconstruction()
        self._vo = visual_odometry.VisualOdometry()
        print("Start reconstruction ...")
        try:
            for i, (keypoints, descriptors) in enumerate(self._frame_reader):
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
        except KeyboardInterrupt:
            print("Keyboard interrupted.")
        finally:
            self._reconstruction.dump(self._output_path)
