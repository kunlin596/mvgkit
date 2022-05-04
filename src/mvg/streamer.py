import os
import threading
import uuid
from collections import deque
from pathlib import Path

import cv2

from mvg import features
from mvg.image_processing import Image


class StreamerBase:
    pass


class ImageFileStreamer(StreamerBase):
    def __init__(self, path: str, buffer_size: int = 10):
        self._path = Path(path)
        self._frame_reader = self._get_frame_reader()
        self._frame_buffer = deque([], maxlen=buffer_size)
        self._frame_buffer_lock = threading.Lock()

    def _get_frame_reader(self):
        filenames = sorted(os.listdir(self._path))
        for filename in filenames[::2]:
            # FIXME: only support Kitti images.
            if len(filename) != 14 or not filename.endswith(".png"):
                continue
            filepath = self._path / filename
            print(f" - Reading {filename} ...")
            image = Image(
                id=uuid.uuid4(),
                data=cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB),
                uri=str(filepath),
            )
            self.frame_buffer.append(image)
            yield self.frame_buffer[-1]

    @property
    def frame_buffer(self):
        with self._frame_buffer_lock:
            return self._frame_buffer

    def get(self):
        return next(self._frame_reader)


class FeatureFileStreamer(StreamerBase):
    def __init__(self, path: str):
        self._path = Path(path)
        self._frame_reader = features.get_feature_iter(self._path)

    def get(self):
        return next(self._frame_reader)
