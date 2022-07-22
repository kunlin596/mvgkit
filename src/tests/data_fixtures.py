import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pytest import fixture

from mvgkit.common.camera import CameraMatrix
from mvgkit.estimation.stereo import Fundamental, FundamentalOptions
from mvgkit.features import SIFT, Matcher
from mvgkit.image_processing import Image


@dataclass
class StereoDataPack:
    image_L: np.ndarray
    image_R: np.ndarray
    points_L: np.ndarray
    points_R: np.ndarray
    camera_matrix: Optional[CameraMatrix] = None
    manual_points_L: Optional[np.ndarray] = None
    manual_points_R: Optional[np.ndarray] = None
    F_RL: Optional[np.ndarray] = None
    inlier_indices: Optional[np.ndarray] = None


@fixture
def leuven_stereo_data_pack(data_root_path):
    fundamental_root_path = data_root_path / "fundamental"
    with open(fundamental_root_path / "meta.json") as f:
        meta = json.load(f)

    assert meta.get("manualPointsL") is not None
    manual_points_L = np.asarray(meta["manualPointsL"], dtype=float)
    assert meta.get("manualPointsR") is not None
    manual_points_R = np.asarray(meta["manualPointsR"], dtype=float)

    image_L = Image.from_file(str(fundamental_root_path / meta["left"])).data
    image_R = Image.from_file(str(fundamental_root_path / meta["right"])).data
    camera_matrix = CameraMatrix.from_matrix(np.reshape(meta["K"], (3, 3)))

    keypoints_L, descriptors_L = SIFT.detect(image_L)
    keypoints_R, descriptors_R = SIFT.detect(image_R)
    query_indices, train_indices = Matcher.match(
        descriptors1=descriptors_L, descriptors2=descriptors_R
    )
    points_L = np.asarray([kp.pt for kp in keypoints_L[query_indices]])
    points_R = np.asarray([kp.pt for kp in keypoints_R[train_indices]])

    fundamental = Fundamental(
        FundamentalOptions(atol=20.0), x_L=manual_points_L, x_R=manual_points_R
    )
    F_RL = fundamental.get_F_RL().astype(np.float64)
    inlier_indices = fundamental.get_inlier_indices()

    return StereoDataPack(
        image_L=image_L,
        image_R=image_R,
        manual_points_L=manual_points_L,
        manual_points_R=manual_points_R,
        points_L=points_L,
        points_R=points_R,
        camera_matrix=camera_matrix,
        F_RL=F_RL,
        inlier_indices=list(inlier_indices),
    )


@fixture
def aloe_stereo_data_pack(data_root_path):
    root_path = data_root_path / "stereo" / "aloe"
    with open(root_path / "meta.json") as f:
        meta = json.load(f)
    image_L = Image.from_file(str(root_path / meta["left"])).resize(0.4).data
    image_R = Image.from_file(str(root_path / meta["right"])).resize(0.4).data

    keypoints_L, descriptors_L = SIFT.detect(image_L)
    keypoints_R, descriptors_R = SIFT.detect(image_R)
    query_indices, train_indices = Matcher.match(
        descriptors1=descriptors_L, descriptors2=descriptors_R
    )
    points_L = np.asarray([kp.pt for kp in keypoints_L[query_indices]])
    points_R = np.asarray([kp.pt for kp in keypoints_R[train_indices]])
    fundamental = Fundamental(FundamentalOptions())
    fundamental(x_L=points_L, x_R=points_R)
    F_RL = fundamental.get_F_RL()
    inlier_indices = fundamental.get_inlier_indices()

    return StereoDataPack(
        image_L=image_L,
        image_R=image_R,
        manual_points_L=None,
        manual_points_R=None,
        points_L=points_L,
        points_R=points_R,
        camera_matrix=None,
        F_RL=F_RL,
        inlier_indices=list(inlier_indices),
    )


@fixture
def book_stereo_data_pack(data_root_path):
    root_path = data_root_path / "stereo" / "book"
    with open(root_path / "meta.json") as f:
        meta = json.load(f)
    image_L = Image.from_file(root_path / meta["left"]).data
    image_R = Image.from_file(root_path / meta["right"]).data
    F_RL = np.asarray(meta["F_RL"])

    points_L = np.asarray(meta["points_L"])
    points_R = np.asarray(meta["points_R"])
    inlier_mask = np.asarray(meta["inlier_mask"], dtype=bool)

    return StereoDataPack(
        image_L=image_L,
        image_R=image_R,
        manual_points_L=None,
        manual_points_R=None,
        points_L=points_L,
        points_R=points_R,
        camera_matrix=None,
        F_RL=F_RL,
        inlier_indices=np.nonzero(inlier_mask)[0],
    )
