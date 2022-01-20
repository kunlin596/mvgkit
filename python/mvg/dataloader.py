#!/usr/bin/env python3


import os
import PIL
from pathlib import Path

from mvg.image_processing import Image


class ImageFileReader:
    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self):
        for filename in os.listdir(self._path.absolute()):
            if not filename.split(".")[-1] in ["png", "jpg"]:
                continue
            abspath = self._path / filename
            from IPython import embed

            embed()
            image_data = PIL.read(str(abspath))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            # TODO(kun): try to read exif
            yield Image(image_data)
