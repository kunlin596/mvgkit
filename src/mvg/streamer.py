from pathlib import Path
from mvg import features
import os
import cv2


class StreamerBase:
    pass


class ImageFileStreamer(StreamerBase):
    def __init__(self, path: str):
        self._path = Path(path)
        self._frame_reader = self._get_frame_reader()

    def _get_frame_reader(self):
        for filename in os.listdir(self._path):
            # FIXME: only support Kitti images.
            if len(filename) != 14 or not filename.endswith(".png"):
                continue

            filepath = self._path / filename
            print(f" - Reading {filename} ...")
            yield cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)

    def get(self):
        return next(self._frame_reader)


class FeatureFileStreamer(StreamerBase):
    def __init__(self, path: str):
        self._path = Path(path)
        self._frame_reader = features.get_feature_iter(self._path)

    def get(self):
        return next(self._frame_reader)
