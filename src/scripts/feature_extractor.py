#!/usr/bin/env python3

import argparse
import copyreg
import os
import pickle
from pathlib import Path

import cv2
import tqdm
from mvg import features


def _patch_keypoint_pickle():
    """
    This util function is for fixing the pickle issue of cv2.KeyPoint.

    NOTE:
        C++ constructor
        KeyPoint(
            float x,
            float y,
            float _size,
            float _angle=-1,
            float _response=0,
            int _octave=0,
            int _class_id=-1
        )
    """

    def _pickle_keypoint(keypoint):
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )

    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)


def _main(input_dir, output_dir, feature_type):
    _patch_keypoint_pickle()

    feature_cls = getattr(features, feature_type)
    extractor = feature_cls()

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm.tqdm(os.listdir(input_dir), desc="Computing"):
        if Path(filename).suffix != ".png":
            continue
        image_path = input_dir / filename
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        feats = extractor.detect(image)

        output_file_path = output_dir / Path(filename).with_suffix(".feat")
        with open(output_file_path, "wb") as f:
            pickle.dump(feats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--feature", "-f", type=str, default="sift", choices=["sift", "orb"])

    options = parser.parse_args()

    _main(Path(options.input), Path(options.output), options.feature.upper())
