"""This modules implements a variety of SfM algorithms
"""
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from mvg import features, streamer
from mvg.basic import SE3
from mvg.camera import Camera
from mvg.mapping.common import frame, visual_odometry
from mvg.mapping.common.reconstruction import Reconstruction
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

_N_CAMERA_PARAM = 6
_N_POINT_PARAM = 3


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
                self._reconstruction = Reconstruction()

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

    @staticmethod
    def _residual(x, reconstruction: Reconstruction, camera: Camera):
        # FIXME: This is a quick implementation, need to optimize it.
        poses_G, points3d_G = IncrementalSFM._decompose_parameters(x, len(reconstruction.frames))
        residuals = []
        for landmark_index, landmark_G in enumerate(reconstruction.landmarks_G):
            point3d_G = points3d_G[landmark_index]
            for frame_id, observation in landmark_G.observations.items():
                pose_G = poses_G[reconstruction.get_frame_index_from_id(frame_id)]
                reprojected_image_point = camera.project_points([pose_G.inv() @ point3d_G])[0]
                residuals.append(reprojected_image_point - observation["observation"])
        return np.asarray(residuals).reshape(-1)

    @staticmethod
    def _compose_parameters(reconstruction: Reconstruction):
        return np.r_[
            np.reshape([frame.pose_G.as_rotvec_pose() for frame in reconstruction.frames], -1),
            np.reshape([landmark_G.pose_G.t for landmark_G in reconstruction.landmarks_G], -1),
        ]

    @staticmethod
    def _decompose_parameters(x, num_frames):
        poses_G = [
            SE3.from_rotvec_pose(x[i : i + _N_CAMERA_PARAM])
            for i in range(0, num_frames * _N_CAMERA_PARAM, _N_CAMERA_PARAM)
        ]
        points3d_G = x[num_frames * _N_CAMERA_PARAM :].reshape(-1, _N_POINT_PARAM)
        return (poses_G, points3d_G)

    @staticmethod
    def _get_jac_sparsity(reconstruction: Reconstruction):
        num_landmarks = len(reconstruction.landmarks_G)
        num_frames = len(reconstruction.frames)
        n = num_frames * _N_CAMERA_PARAM + num_landmarks * _N_POINT_PARAM
        A = []
        frame_indices = []
        landmark_indices = []
        for landmark_index, landmark_G in enumerate(reconstruction.landmarks_G):
            for frame_id, _ in landmark_G.observations.items():
                frame_indices.append(reconstruction.get_frame_index_from_id(frame_id))
                landmark_indices.append(landmark_index)
        frame_indices = np.asarray(frame_indices, dtype=np.int32)
        landmark_indices = np.asarray(landmark_indices, dtype=np.int32)
        m = frame_indices.size * 2
        i = np.arange(frame_indices.size)
        A = lil_matrix((m, n), dtype=int)
        for j in range(_N_CAMERA_PARAM):
            A[2 * i, frame_indices * _N_CAMERA_PARAM + j] = 1
            A[2 * i + 1, frame_indices * _N_CAMERA_PARAM + j] = 1
        for j in range(_N_POINT_PARAM):
            A[2 * i, num_frames * _N_CAMERA_PARAM + landmark_indices * _N_POINT_PARAM + j] = 1
            A[2 * i + 1, num_frames * _N_CAMERA_PARAM + landmark_indices * _N_POINT_PARAM + j] = 1
        return A

    def bundle_adjustment(self):
        result = least_squares(
            fun=self._residual,
            x0=self._compose_parameters(self._reconstruction),
            jac_sparsity=self._get_jac_sparsity(self._reconstruction),
            kwargs=dict(reconstruction=self._reconstruction, camera=self._camera),
            x_scale="jac",
            ftol=1e-4,
            method="trf",
            loss="huber",
            verbose=1,
        )
        if result["success"]:
            poses_G, points_G = self._decompose_parameters(
                result["x"], len(self._reconstruction.frames)
            )
            for i, f in enumerate(self._reconstruction.frames):
                f.pose_G = poses_G[i]
            for i, landmark_G in enumerate(self._reconstruction.landmarks_G):
                landmark_G.pose_G.t = points_G[i]
            observation_residuals = np.linalg.norm(result["fun"].reshape(-1, 2), axis=1)
            res_index = 0
            to_preserve = []
            for landmark_index, _ in enumerate(self._reconstruction.landmarks_G):
                dists = []
                for _ in enumerate(landmark_G.observations.values()):
                    dists.append(observation_residuals[res_index])
                    res_index += 1
                if np.asarray(dists).mean() < 2.0 and np.asarray(dists).std() < 2.0:
                    to_preserve.append(landmark_index)
            print(
                f" - {'# preversed landmarks':35s}: {len(to_preserve)} / {len(self._reconstruction.landmarks_G)}."
            )
            # FIXME
            self._reconstruction._landmarks_G = np.asarray(self._reconstruction._landmarks_G)[
                to_preserve
            ].tolist()

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
                if i > 1:
                    self.bundle_adjustment()
                i += 1

        except KeyboardInterrupt:
            print("Keyboard interrupted.")
