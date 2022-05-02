"""This module implements visual odometry classes
"""
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np
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

        # 2D features
        num_features: int
        scaling_factor: float
        level_pyramid: int

        # 2D feature matching
        matching_ratio: float

        # VO state machine
        max_num_tracking_failures: int
        num_inliers_per_frame_pair: int
        min_rotation_per_frame_pair: float
        min_translation_per_frame_pair: float

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
        self._params = params
        self._state = self.State.InitializingFirst
        self._num_failures = 0

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
        P1 = camera_matrix @ SE3.from_rotmat_tvec(np.eye(3), np.zeros(3)).as_augmented_matrix()

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
        for i, P2 in enumerate(P2_candidates):
            points3d_L = triangulate(P1, P2, image_points_L, image_points_R)
            # FIXME: determine min depth correctly.
            inlier_mask = points3d_L[:, 2] > 1.0
            num_valid_points = np.count_nonzero(inlier_mask)

            if num_valid_points > max_num_valid_points:
                max_num_valid_points = num_valid_points
                best_T_RL = T_RL_candidates[i]
                best_points3d_L = points3d_L

        return best_T_RL, best_points3d_L

    @staticmethod
    def _initialize_reconstruction(reconstruction: Reconstruction):
        """Initialize reconstruction using the first two frames."""
        assert len(reconstruction.frames) == 2, "Need 2 frames to initialize map!"

        f1 = reconstruction.frames[0]
        f2 = reconstruction.frames[1]

        keypoints_L = f1.keypoints
        keypoints_R = f2.keypoints

        descriptors_L = f1.descriptors
        descriptors_R = f2.descriptors

        query_indices, train_indices = Matcher.match(descriptors_L, descriptors_R)

        matched_points_L = _keypoints_to_ndarray(keypoints_L[query_indices])
        matched_points_R = _keypoints_to_ndarray(keypoints_R[train_indices])

        F_RL, _ = Fundamental.compute(x_L=matched_points_L, x_R=matched_points_R)
        # NOTE: Current implementation only support same camera.
        E_RL = f1.camera.K.as_matrix().T @ F_RL @ f2.camera.K.as_matrix()
        R1_RL, R2_RL, t_R = decompose_essential_matrix(E_RL=E_RL)

        rel_pose_RL, points3d_L = VisualOdometry._disambiguate_camera_pose(
            R1_RL=R1_RL,
            R2_RL=R2_RL,
            t_R=t_R,
            camera_matrix=f1.camera.K.as_matrix(),
            image_points_L=matched_points_L,
            image_points_R=matched_points_R,
        )

        ref_image_points = f1.camera.project_points(points3d_L)
        is_x_in_view = (0.0 < ref_image_points[:, 0]) & (ref_image_points[:, 0] < 1226.0)
        is_y_in_view = (0.0 < ref_image_points[:, 1]) & (ref_image_points[:, 1] < 370.0)
        is_points_in_view = is_x_in_view & is_y_in_view

        f2.pose_G = f1.pose_G @ rel_pose_RL.inv()

        # TODO: Find ouf the best k keypoints
        filtered_descriptors = descriptors_L[query_indices][is_points_in_view]
        for i, point3d_L in enumerate(points3d_L[is_points_in_view]):

            if point3d_L[-1] < 0.01 or point3d_L[-1] > 300.0:
                # FIXME: negative points should not exist.
                continue

            reconstruction.add_landmark(
                Landmark(
                    id=uuid.uuid4(),
                    pose_G=SE3.from_rotvec_pose(np.r_[0.0, 0.0, 0.0, point3d_L]),
                    descriptor=filtered_descriptors[i],
                    frame_id=f1.id,
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
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)

        is_ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=0.5,
            iterationsCount=2000,
        )

        if not is_ok:
            raise Exception("Solve PnP failed!")

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
    def _add_frame(reconstruction: Reconstruction, frame: Frame):
        reconstruction.add_frame(frame)

        # Get the latest 2 key frames.
        cur_frame = reconstruction.frames[-1]
        ref_frame = reconstruction.frames[-2]

        # Get all points and descriptors from the existing reconstruction.
        map_descriptors = reconstruction.get_descriptors()
        landmark_positions_G = reconstruction.get_landmark_positions_G()

        # Filter points in view.
        ref_image_points = ref_frame.camera.project_points(
            ref_frame.pose_G.inv() @ landmark_positions_G
        )

        # FIXME: store image size in camera.
        is_x_in_view = (0.0 < ref_image_points[:, 0]) & (ref_image_points[:, 0] < 1226.0)
        is_y_in_view = (0.0 < ref_image_points[:, 1]) & (ref_image_points[:, 1] < 370.0)
        is_points_in_view = is_x_in_view & is_y_in_view

        filtered_map_descriptors = map_descriptors[is_points_in_view]
        filtered_landmark_positions_G = landmark_positions_G[is_points_in_view]

        # Estimate new key frame pose.
        query_indices, train_indices = Matcher.match(
            filtered_map_descriptors, cur_frame.descriptors
        )
        object_points_L = ref_frame.pose_G.inv() @ filtered_landmark_positions_G[query_indices]

        rel_pose_RL = VisualOdometry._solve_pnp(
            object_points=object_points_L,
            image_points=_keypoints_to_ndarray(cur_frame.keypoints[train_indices]),
            camera_matrix=frame.camera.K.as_matrix(),
        )

        # Update new frame pose.
        rel_pose_LR = rel_pose_RL.inv()
        cur_frame.pose_G = ref_frame.pose_G @ rel_pose_LR

        # Triangulate new map points.
        P_L = np.hstack([cur_frame.camera.K.as_matrix(), np.zeros((3, 1))])
        P_R = ref_frame.camera.K.as_matrix() @ rel_pose_RL.as_augmented_matrix()
        new_query_indices, train_indices = Matcher.match(
            ref_frame.descriptors, cur_frame.descriptors
        )
        new_query_keypoints = _keypoints_to_ndarray(ref_frame.keypoints[new_query_indices])
        new_train_keypoints = _keypoints_to_ndarray(cur_frame.keypoints[train_indices])
        new_points3d_L = triangulate(P_L, P_R, new_query_keypoints, new_train_keypoints)

        # Point cloud simplification.
        # TODO: Find better way to do this.
        new_points3d_G = ref_frame.pose_G @ new_points3d_L
        map_tree = KDTree(filtered_landmark_positions_G)
        dists = np.asarray([map_tree.query(lm_pose_G)[0] for lm_pose_G in new_points3d_G])
        min_dist = min(np.percentile(dists, 90) * 0.5, 30.0)

        # Update new 3d points into reconstruction.
        for i, new_point3d_G in enumerate(new_points3d_G):
            if new_point3d_G[-1] < 0.1 or np.any(np.abs(new_point3d_G) > 300.0):
                continue

            if dists[i] < min_dist:
                continue

            lm_pose_G = SE3.from_rotvec_pose(np.r_[0.0, 0.0, 0.0, new_point3d_G])

            reconstruction.add_landmark(
                Landmark(
                    id=uuid.uuid4(),
                    pose_G=lm_pose_G,
                    descriptor=ref_frame.descriptors[new_query_indices][i],
                    frame_id=ref_frame.id,
                )
            )

        # Validate new frame pose. TODO: add more validation.
        translation = np.linalg.norm(rel_pose_LR.t)
        if translation > 10.0:
            print(f" - translation is too large: {translation:7.3f}.")
            return False
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
