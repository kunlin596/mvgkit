import uuid
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from mvg.basic import SE3
from mvg.camera import Camera
from mvg.image_processing import Image


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

    def __repr__(self) -> str:
        num_points3d = len(self.points3d) if self.points3d is not None else None
        return (
            f"Frame(id={self.id}, #keypoints={len(self.keypoints)}, "
            f"#points3d={num_points3d}, pose_G={self.pose_G})"
        )

    def has_point(self, point: np.ndarray):
        # TODO
        return True
