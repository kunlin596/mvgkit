import copyreg
import pickle
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import plyfile
from mvg.basic import SE3
from mvg.mapping.frame import Frame


@dataclass
class Landmark:
    id: uuid.UUID
    pose_G: SE3
    descriptor: np.ndarray
    frame_id: uuid.UUID
    num_matches: Optional[int] = -1  # TODO
    num_observations: Optional[int] = -1  # TODO


def _patch_keypoint_pickle():
    """
    This util function is for fixing the pickle issue of cv2.KeyPoint.

    NOTE:
        C++ constructor
        KeyPoint(
            float x,
            float y,
            float _size,
            float _angle=-1,
            float _response=0,
            int _octave=0,
            int _class_id=-1
        )
    """

    def _pickle_keypoint(keypoint):
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )

    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)


class Reconstruction:
    def __init__(self):
        # TODO: Use pandas.DataFrame
        self._landmarks: List[Landmark] = list()
        self._frames: List[Frame] = list()

    def add_frame(self, frame: Frame):
        self._frames.append(frame)

    def add_landmark(self, landmark: Landmark):
        self._landmarks.append(landmark)

    def get_descriptors(self):
        return np.vstack([landmark.descriptor for landmark in self._landmarks])

    def get_landmark_positions_G(self):
        # TODO: Let it return the pose vector.
        return np.vstack([landmark.pose_G.t for landmark in self._landmarks])

    @property
    def frames(self) -> np.ndarray:
        return np.asarray(self._frames)

    @property
    def landmarks(self) -> List[Landmark]:
        return self._landmarks

    def dump(self, path: Path):
        points = self.get_landmark_positions_G()
        filepath = path / "reconstruction.ply"
        print(f"Writing to {filepath}.")
        vertices = np.array(
            [tuple(e) for e in points], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        )
        el = plyfile.PlyElement.describe(vertices, "vertex")
        plyfile.PlyData([el], text=True).write(filepath)

    @staticmethod
    def load(path) -> "Reconstruction":
        with open(path, "rb") as file:
            return pickle.load(file)
