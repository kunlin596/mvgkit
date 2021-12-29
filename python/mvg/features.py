#!/usr/bin/env python3
"""This module includes various common image features."""
from typing import List, Optional, Dict
import cv2
import numpy as np


class ORB:
    _detector = None

    def __init__(self):
        self._detector = cv2.ORB_create()

    def __call__(self, image: np.ndarray):
        return self._detector.detectAndCompute(image, None)

    @staticmethod
    def detect(image: np.ndarray):
        return ORB()(image)

    @staticmethod
    def draw(image: np.ndarray, keypoints: list):
        return cv2.drawKeypoints(image.copy(), keypoints, None)


class SIFT:
    _detector = None

    def __init__(self, options: Optional[dict] = None):
        self._detector = cv2.SIFT_create(nfeatures=20000)

    def __call__(self, image: np.ndarray):
        return self._detector.detectAndCompute(image, None)

    @staticmethod
    def detect(image: np.ndarray):
        return SIFT()(image)

    @staticmethod
    def draw(image: np.ndarray, keypoints: list):
        return cv2.drawKeypoints(image.copy(), keypoints, None)


class Matcher:
    _matcher = None
    _index_params = None
    _search_params = None

    FLANN_INDEX_KDTREE = 0

    def __init__(self, index_params: Optional[Dict] = None, search_params: Optional[Dict] = None):
        self._index_params = index_params
        if self._index_params is None:
            self._index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self._search_params = search_params
        if self._search_params is None:
            self._search_params = dict(checks=50)

        self._matcher = cv2.FlannBasedMatcher(self._index_params, self._search_params)

    def __call__(self, descriptors1, descriptors2, k: Optional[int] = None):
        if k is None:
            k = 2
        return self._matcher.knnMatch(descriptors1, descriptors2, k=k)

    @staticmethod
    def match(
        *,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        index_params: Optional[Dict] = None,
        search_params: Optional[Dict] = None,
        k: Optional[int] = None,
    ):
        return Matcher(index_params, search_params)(
            descriptors1.astype(np.float32), descriptors2.astype(np.float32), k
        )

    @staticmethod
    def get_matched_points(keypoints1, keypoints2, matches, dist_threshold=0.8):
        points1 = []
        points2 = []
        good_matches = []
        for m, n in matches:
            if m.distance < dist_threshold * n.distance:
                good_matches.append(m)
                points1.append(keypoints1[m.queryIdx].pt)
                points2.append(keypoints2[m.trainIdx].pt)
        return np.asarray(points1), np.asarray(points2), good_matches

    @staticmethod
    def draw(
        *,
        image1: np.ndarray,
        keypoints1: List[cv2.KeyPoint],
        image2: np.ndarray,
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        draw_params=None,
    ):
        return cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None)
