#!/usr/bin/env python3


import os
import PIL.Image
import PIL.ExifTags
from pathlib import Path

import numpy as np

from mvg.image_processing import Image


class ImageFileReader:
    def __init__(self, path: Path) -> None:
        self._path = path

    @staticmethod
    def _extract_exif(image: PIL.Image):
        exif = image.getexif()
        meta = dict()
        for k, w in exif.items():
            meta[PIL.ExifTags.TAGS.get(k, k)] = w
        return meta

    @staticmethod
    def readone(path: Path):
        image = PIL.Image.open(path)
        meta = ImageFileReader._extract_exif(image)
        return Image(data=np.array(image), meta=meta)

    def read(self):
        for filename in os.listdir(self._path.absolute()):
            if not filename.split(".")[-1] in ["png", "jpg"]:
                continue
            yield self.readone(self._path / filename)
