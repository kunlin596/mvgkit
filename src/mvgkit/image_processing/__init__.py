#!/usr/bin/env python3

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def resize(image, ratio=0.5):
    return cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))


@dataclass
class Image:
    id: uuid.UUID
    data: np.ndarray
    timestamp: float = -1.0  # TODO
    uri: str = ""
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None

    def __post_init__(self):
        self._size = np.asarray(self.data.shape)[[1, 0]]

    @property
    def size(self):
        return self._size

    @staticmethod
    def from_file(path: Path):
        image_data = cv2.imread(str(path))
        # TODO(kun): Check image channel
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return Image(id=uuid.uuid4(), data=image_data, uri=str(path))

    def resize(self, ratio=0.5):
        # FIXME: should the resized image be considered as a new image?
        return Image(id=self.id, data=resize(self.data, ratio), uri=self.uri)
