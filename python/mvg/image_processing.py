#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional
import cv2
from pathlib import Path
import numpy as np


def resize(image, ratio=0.5):
    return cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))


@dataclass
class Image:
    data: np.ndarray
    meta: Optional[dict] = None

    @staticmethod
    def from_file(path: Path):
        image_data = cv2.imread(str(path))
        # TODO(kun): Check image channel
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return Image(image_data)

    def resize(self, ratio=0.5):
        return Image(resize(self.data, ratio))
