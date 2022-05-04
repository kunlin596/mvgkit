import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plyfile
from mvg.basic import SE3
from mvg.mapping.common.frame import Frame


@dataclass
class Landmark:
    id: uuid.UUID
    pose_G: SE3
    descriptor: np.ndarray  # TODO: Compute a representative from the observations
    observations: Dict[int, Tuple[Frame, Dict]]


class Reconstruction:
    def __init__(self):
        # TODO: Use pandas.DataFrame
        self._landmarks_G: List[Landmark] = list()
        self._frames: List[Frame] = list()
        self._frame_index_lookup = dict()

    def add_frame(self, frame: Frame):
        self._frames.append(frame)
        self._frame_index_lookup[frame.id] = len(self._frames) - 1

    def get_frame_index_from_id(self, frame_id: uuid.UUID):
        return self._frame_index_lookup.get(frame_id)

    def add_landmark_G(self, landmark: Landmark):
        self._landmarks_G.append(landmark)

    def extend_landmarks_G(self, new_landmarks_G: List[Landmark]):
        self._landmarks_G.extend(new_landmarks_G)

    def get_descriptors(self):
        if len(self.landmarks_G):
            return np.vstack([landmark.descriptor for landmark in self.landmarks_G])
        return []

    def get_landmark_positions_G(self):
        # TODO: Let it return the pose vector.
        if len(self.landmarks_G):
            return np.vstack([landmark.pose_G.t for landmark in self.landmarks_G])
        return []

    @property
    def frames(self) -> np.ndarray:
        return np.asarray(self._frames)

    @property
    def landmarks_G(self) -> np.ndarray:
        return np.asarray(self._landmarks_G)

    def dump(self, path: Path):
        points = self.get_landmark_positions_G()
        model_filepath = path / "reconstruction.ply"
        print(f"Writing model to {model_filepath}.")
        vertices = np.array(
            [tuple(e) for e in points], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        )
        el = plyfile.PlyElement.describe(vertices, "vertex")
        plyfile.PlyData([el], text=True).write(model_filepath)

        trajectory_filepath = path / "trajectory.txt"
        print(f"Writing trajectory to {trajectory_filepath}.")
        poses = np.vstack([frame.pose_G.as_rotvec_pose() for frame in self._frames])
        np.savetxt(trajectory_filepath, poses)
