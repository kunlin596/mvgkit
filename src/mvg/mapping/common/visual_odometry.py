"""This module implements visual odometry classes
"""
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np
from mvg.algorithms.optical_flow import OpticalFlowLK
from mvg.basic import SE3
from mvg.features import Matcher
from mvg.mapping.common.frame import Frame
from mvg.mapping.common.reconstruction import Landmark, Reconstruction
from mvg.stereo import Fundamental, decompose_essential_matrix, triangulate
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


def _keypoints_to_ndarray(keypoints):
    """NOTE: create better IO for keypoints"""
    return np.asarray([kp.pt for kp in keypoints], dtype=np.float32)


class VisualOdometry:
    """This class implements a consecutive image feature based visual odometry."""

    @dataclass
    class Parameters:
        """Parameter class for visual odometry."""

        # TODO
        # 2D features
        # num_features: int
        # scaling_factor: float
        # level_pyramid: int

        # 2D feature matching
        # matching_ratio: float

        # VO state machine
        # max_num_tracking_failures: int
        # num_inliers_per_frame_pair: int
        # min_rotation_per_frame_pair: float
        # min_translation_per_frame_pair: float

        debug: bool = False

    class State(IntEnum):
        """VO States."""

        InitializingFirst = 0
        InitializingSecond = 1
        Normal = 2
        Failed = 3

    def __init__(self, params: Optional[Parameters] = None):
        if params is None:
            # TODO: add parameters
            pass
        self._params = self.Parameters(**params)
        self._state = self.State.InitializingFirst
        self._num_failures = 0

    @property
    def params(self):
        return self._params

    @staticmethod
    def _disambiguate_camera_pose(
        R1_RL: Rotation,
        R2_RL: Rotation,
        t_R: np.ndarray,
        camera_matrix: np.ndarray,
        image_points_L: np.ndarray,
        image_points_R: np.ndarray,
    ):
        """Find the correct camera pose from result from essential matrix decomposition."""
        P1 = np.hstack([camera_matrix, np.zeros((3, 1))])
        T_RL_candidates = [
            SE3.from_rotmat_tvec(R1_RL, t_R),
            SE3.from_rotmat_tvec(R1_RL, -t_R),
            SE3.from_rotmat_tvec(R2_RL, t_R),
            SE3.from_rotmat_tvec(R2_RL, -t_R),
        ]

        P2_candidates = [camera_matrix @ T_RL.as_augmented_matrix() for T_RL in T_RL_candidates]
        max_num_valid_points = -1
        best_T_RL = None
        best_points3d_L = None
        best_inlier_mask = None
        for i, P2 in enumerate(P2_candidates):
            points3d_L = triangulate(P1, P2, image_points_L, image_points_R)
            # FIXME: determine min depth correctly.
            inlier_mask = points3d_L[:, 2] > 1.0
            num_valid_points = np.count_nonzero(inlier_mask)

            if num_valid_points > max_num_valid_points:
                max_num_valid_points = num_valid_points
                best_T_RL = T_RL_candidates[i]
                best_points3d_L = points3d_L
                best_inlier_mask = inlier_mask

        return best_T_RL, best_points3d_L, best_inlier_mask

    @staticmethod
    def _estimate_stereo_pose(frame_R: Frame, frame_C: Frame):
        keypoints_R = frame_R.keypoints
        keypoints_C = frame_C.keypoints

        descriptors_R = frame_R.descriptors
        descriptors_C = frame_C.descriptors

        query_indices, train_indices = Matcher.match(descriptors_R, descriptors_C)

        matched_points_R = _keypoints_to_ndarray(keypoints_R[query_indices])
        matched_points_C = _keypoints_to_ndarray(keypoints_C[train_indices])

        F_CR, fundamental_inlier_mask = Fundamental.compute(
            x_L=matched_points_R, x_R=matched_points_C
        )
        E_CR = frame_R.camera.K.as_matrix().T @ F_CR @ frame_C.camera.K.as_matrix()
        R1_CR, R2_CR, t_C = decompose_essential_matrix(E_RL=E_CR)

        rel_pose_CR, points3d_R, inlier_mask_R = VisualOdometry._disambiguate_camera_pose(
            R1_RL=R1_CR,
            R2_RL=R2_CR,
            t_R=t_C,
            camera_matrix=frame_R.camera.K.as_matrix(),
            image_points_L=matched_points_R[fundamental_inlier_mask],
            image_points_R=matched_points_C[fundamental_inlier_mask],
        )

        return (
            rel_pose_CR,
            points3d_R[inlier_mask_R],
            descriptors_R[query_indices][fundamental_inlier_mask][inlier_mask_R],
        )

    def _initialize_reconstruction(self, reconstruction: Reconstruction):
        """Initialize reconstruction using the first two frames."""
        assert len(reconstruction.frames) == 2, "Need two frames to initialize map!"

        frame_R = reconstruction.frames[-2]
        frame_C = reconstruction.frames[-1]
        rel_pose_CR, points3d_R, descriptors_R = self._estimate_stereo_pose(frame_R, frame_C)
        is_in_range = self._get_points3d_range_mask(points3d_R)
        points3d_R = points3d_R[is_in_range]
        descriptors_R = descriptors_R[is_in_range]
        frame_C.pose_G = frame_R.pose_G @ rel_pose_CR.inv()
        frame_R.points3d = points3d_R
        points3d_G = frame_R.pose_G @ points3d_R
        for i, point3d_G in enumerate(points3d_G):
            reconstruction.add_landmark_G(
                Landmark(
                    id=uuid.uuid4(),
                    pose_G=SE3.from_rotvec_pose(np.r_[0.0, 0.0, 0.0, point3d_G]),
                    descriptor=descriptors_R[i],
                    frame_id=frame_R.id,
                )
            )

    @staticmethod
    def _solve_pnp(
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        """Solve for the transformation from world frame to camera frame."""

        assert len(object_points) >= 4, "Not enough points for solving pnp, need at least 4."

        is_ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=1.0,
            iterationsCount=2000,
            confidence=0.9,
        )

        if not is_ok:
            raise Exception("Solve PnP failed!")

        print(f" - {'PnP inliers size':35s}: {len(inliers):5d}.")

        # NOTE: no need, but included just to make it look nicer on terminal.
        inliers = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=object_points[inliers],
            imagePoints=image_points[inliers],
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
        )
        pose = SE3.from_rotvec_pose(np.r_[rvec.reshape(-1), tvec.reshape(-1)])
        return pose

    @staticmethod
    def _get_image_point_visibility_mask(image_points):
        # FIXME: store image size in camera.
        is_x_in_view = (0.0 < image_points[:, 0]) & (image_points[:, 0] < 1226.0)
        is_y_in_view = (0.0 < image_points[:, 1]) & (image_points[:, 1] < 370.0)
        return is_x_in_view & is_y_in_view

    @staticmethod
    def _get_points3d_range_mask(points3d, max_val=20.0):
        is_x_in_range = np.abs(points3d[:, 0]) < max_val
        is_y_in_range = np.abs(points3d[:, 1]) < max_val
        is_z_in_range = (0.01 < points3d[:, 2]) & (points3d[:, 2] < max_val)
        return is_x_in_range & is_y_in_range & is_z_in_range

    @staticmethod
    def _filter_visible_landmarks_in_frame_R(landmarks_G: np.ndarray, frame_R: Frame):
        """Filter out the landmarks in frame (G) that is visible in frame (R)."""
        landmark_positions_G = np.vstack([lm.pose_G.t for lm in landmarks_G])
        front_mask = landmark_positions_G[:, -1] > 1.0
        map_points3d_R = frame_R.pose_G.inv() @ landmark_positions_G[front_mask]
        map_image_points_R = frame_R.camera.project_points(map_points3d_R)
        is_in_view = VisualOdometry._get_image_point_visibility_mask(map_image_points_R)
        return landmarks_G[front_mask][is_in_view]

    @staticmethod
    def _filter_consistent_landmarks_in_frame_R(
        landmarks_G: np.ndarray,
        frame_R: Frame,
        reprojection_error_threshold=2.0,
    ):
        """Filter out the landmarks that have consistent descriptors in frame (R).

        Match the reprojected landmarks in frame (R) and match it with the detected ones.
        """
        landmark_positions_G = np.vstack([lm.pose_G.t for lm in landmarks_G])
        map_descriptors_G = np.vstack([landmark_G.descriptor for landmark_G in landmarks_G])
        query_indices, train_indices = Matcher.match(map_descriptors_G, frame_R.descriptors)
        map_points3d_R = frame_R.pose_G.inv() @ landmark_positions_G
        map_image_points_R = frame_R.camera.project_points(map_points3d_R)
        matched_map_image_points_R = map_image_points_R[query_indices]
        keypoints_R = _keypoints_to_ndarray(frame_R.keypoints[train_indices])
        reprojection = np.linalg.norm(matched_map_image_points_R - keypoints_R, axis=1)
        reprojection_mask = reprojection < reprojection_error_threshold
        return landmarks_G[query_indices][reprojection_mask]

    @staticmethod
    def _solve_new_pose(landmarks_G: np.ndarray, frame_R: Frame, frame_C: Frame):
        if len(landmarks_G) < 4:
            raise Exception("Not enough landmarks!")
        map_descriptors_G = np.vstack([landmark_G.descriptor for landmark_G in landmarks_G])
        query_indices, train_indices = Matcher.match(map_descriptors_G, frame_C.descriptors)
        map_points3d_G = np.vstack([lm.pose_G.t for lm in landmarks_G[query_indices]])
        map_points3d_R = frame_R.pose_G.inv() @ map_points3d_G
        image_points_C = _keypoints_to_ndarray(frame_C.keypoints[train_indices])
        rel_pose_CR = VisualOdometry._solve_pnp(
            object_points=map_points3d_R,
            image_points=image_points_C,
            camera_matrix=frame_C.camera.K.as_matrix(),
        )
        return rel_pose_CR

    @staticmethod
    def _estimate_new_frame_pose(reconstruction: Reconstruction):
        """Estimate new frame pose using latest two frames in the map.
        Frame (R) is the reference frame.
        frame (C) is the current frame.
        Frame (G) is the global frame.
        """
        frame_R = reconstruction.frames[-2]
        frame_C = reconstruction.frames[-1]
        landmarks_G = reconstruction.landmarks_G
        landmarks_G = VisualOdometry._filter_visible_landmarks_in_frame_R(landmarks_G, frame_R)
        print(f" - {'Visible landmarks in (R)':35s}: {len(landmarks_G):5d}.")
        landmarks_G = VisualOdometry._filter_consistent_landmarks_in_frame_R(landmarks_G, frame_R)
        print(f" - {'Consistent landmarks in (G)':35s}: {len(landmarks_G):5d}.")
        rel_pose_CR = VisualOdometry._solve_new_pose(landmarks_G, frame_R, frame_C)
        print(f" - {'Relative translation (R) to (C)':35s}: {rel_pose_CR.t}.")
        return rel_pose_CR

    @staticmethod
    def _generate_new_landmarks(
        rel_pose_CR: SE3,
        frame_R: Frame,
        frame_C: Frame,
        map_points_G: np.ndarray,
    ):
        # Update new frame pose.
        rel_pose_RC = rel_pose_CR.inv()
        frame_C.pose_G = frame_R.pose_G @ rel_pose_RC
        P_R = np.hstack([frame_R.camera.K.as_matrix(), np.zeros((3, 1))])
        P_C = frame_C.camera.K.as_matrix() @ rel_pose_CR.as_augmented_matrix()
        query_indices, train_indices = Matcher.match(frame_R.descriptors, frame_C.descriptors)
        keypoints_R = _keypoints_to_ndarray(frame_R.keypoints[query_indices])
        keypoints_C = _keypoints_to_ndarray(frame_C.keypoints[train_indices])

        # Using optical flow to cross validate matches.
        predicted_keypoints_C, _ = OpticalFlowLK.track(
            frame_R.image.data, frame_C.image.data, keypoints_R
        )
        flow_mask = np.linalg.norm(predicted_keypoints_C - keypoints_C, axis=1) < 1.0
        keypoints_R = keypoints_R[flow_mask]
        keypoints_C = keypoints_C[flow_mask]

        new_points3d_R = triangulate(P_R, P_C, keypoints_R, keypoints_C)
        is_in_range = VisualOdometry._get_points3d_range_mask(new_points3d_R)
        new_points3d_R = new_points3d_R[is_in_range]
        frame_R.points3d = new_points3d_R
        new_points3d_G = frame_R.pose_G @ new_points3d_R

        new_landmarks_G = []
        new_descriptors = frame_R.descriptors[query_indices][flow_mask][is_in_range]

        tree = KDTree(map_points_G)
        dists, _ = tree.query(new_points3d_G)
        for i in range(len(new_points3d_G)):
            if dists[i] < 0.2:
                continue
            new_point3d_G = new_points3d_G[i]
            new_landmark_pose_G = SE3.from_rotvec_pose(np.r_[np.zeros(3), new_point3d_G])
            new_landmarks_G.append(
                Landmark(
                    id=uuid.uuid4(),
                    pose_G=new_landmark_pose_G,
                    descriptor=new_descriptors[i],
                    frame_id=frame_R.id,
                )
            )
        return new_landmarks_G

    def _add_frame(self, reconstruction: Reconstruction, frame: Frame):
        reconstruction.add_frame(frame)

        # Get the latest two key frames.
        assert len(reconstruction.frames) >= 2
        frame_R = reconstruction.frames[-2]
        frame_C = reconstruction.frames[-1]

        rel_pose_CR = self._estimate_new_frame_pose(reconstruction)
        translation_mag = np.linalg.norm(rel_pose_CR.t)
        if translation_mag > 20.0:
            print(f"Translation too large: {translation_mag:7.3f}!")
            return False
        new_landmarks_G = self._generate_new_landmarks(
            rel_pose_CR, frame_R, frame_C, reconstruction.get_landmark_positions_G()
        )
        reconstruction.extend_landmarks_G(new_landmarks_G)
        return True

    def add_frame(self, reconstruction: Reconstruction, frame: Frame):
        """Add frame into estimation.

        Return False if the tracking failed.
        """
        # TODO: implement a state machine
        if self._state == self.State.InitializingFirst:
            reconstruction.add_frame(frame)
            self._state = self.State.InitializingSecond
            return True

        if self._state == self.State.InitializingSecond:
            reconstruction.add_frame(frame)
            self._initialize_reconstruction(reconstruction)
            self._state = self.State.Normal
            return True

        elif self._state == self.State.Normal:
            return self._add_frame(reconstruction, frame)

        elif self._state == self.State.Failed:
            return False
