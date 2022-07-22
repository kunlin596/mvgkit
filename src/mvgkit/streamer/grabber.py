#!/usr/bin/env python3

from __future__ import annotations

import time

import cv2

from mvgkit.log import get_logger

log = get_logger(logger_name=__name__)


class SimpleGrabber:
    """Simple grabber for snapping images."""

    def __init__(self, *, camid: int = 0):
        self._camid = camid
        log.debug(f"Opening camera {self._camid}...")
        self._cam = cv2.VideoCapture(self._camid)

    def __del__(self):
        self._cam.release()

    def snap(self):
        start_time = time.time()
        log.debug(f"Snapping image on {self._camid}...")
        status, frame = self._cam.read()
        log.debug(f"Snapping finished, took {time.time() - start_time} sec.")
        return frame
