import uuid
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from mvgkit.common.camera import Camera
from mvgkit.common.utils import SE3
from mvgkit.image_processing import Image


@dataclass
class Frame:
    id: uuid.UUID
    timestamp: float
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    camera: Camera
    pose_G: SE3 = SE3.from_rotvec_pose(np.zeros(6))
    image: Optional[Image] = None
    points3d: Optional[np.ndarray] = None

    def has_point(self, point: np.ndarray):
        # TODO
        return True
