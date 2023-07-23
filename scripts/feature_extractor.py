#!/usr/bin/env python3
"""This module implements feature extraction."""
from __future__ import annotations

import copyreg
import os
import pickle
from pathlib import Path

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger as log

from mvgkit import features


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


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--input-dirpath",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="input directory path",
)
@click.option(
    "--output-dirpath",
    "-o",
    type=click.Path(),
    default=".",
    help="output directory path",
)
@click.option(
    "--feature-type",
    "-f",
    required=True,
    type=click.Choice(features.AVAIBLE_FEATURE_EXTRACTORS),
    default="SIFT",
    help="feature type",
)
def extract(input_dirpath: str, output_dirpath: str, feature_type: str):
    _patch_keypoint_pickle()

    feature_cls = getattr(features, feature_type)
    extractor = feature_cls()
    input_dirpath = Path(input_dirpath).absolute()
    log.info(f"Input directory path: {input_dirpath}.")
    log.info(f"Output directory path: {output_dirpath}.")
    log.info(f"Feature type: {feature_type}.")
    os.makedirs(output_dirpath, exist_ok=True)
    output_dirpath = Path(output_dirpath).expanduser().absolute()
    for filename in tqdm.tqdm(os.listdir(input_dirpath), desc="Computing features"):
        if Path(filename).suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        image_path = input_dirpath / filename
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        keypoints, descriptors = extractor.detect(image)

        output_file_path = output_dirpath / Path(filename).with_suffix(".feat")
        with open(output_file_path, "wb") as f:
            pickle.dump((keypoints, descriptors), f)


@click.command()
@click.option(
    "--image-dirpath",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="input directory path",
)
@click.option(
    "--feature-dirpath",
    "-o",
    required=True,
    type=click.Path(exists=True),
    help="output directory path",
)
@click.option(
    "--dump-dirpath",
    "-d",
    type=click.Path(),
    default=Path.cwd() / "dump",
    help="dump directory path",
)
@click.option(
    "--show-mode",
    "-s",
    type=click.Choice(["show", "dump", "both"]),
    default="dump",
    help="show mode, show: show images only, dump: dump images only, both: show and dump images",
)
def show(image_dirpath, feature_dirpath, dump_dirpath, show_mode):
    image_dirpath = Path(image_dirpath).absolute()
    feature_dirpath = Path(feature_dirpath).absolute()
    log.info(f"Input directory path: {image_dirpath}.")
    log.info(f"Output directory path: {feature_dirpath}.")
    image_filenames = sorted(os.listdir(image_dirpath))
    feature_filenames = sorted(os.listdir(feature_dirpath))

    if show_mode in ["dump", "both"]:
        os.makedirs(dump_dirpath, exist_ok=True)
        dump_dirpath = Path(dump_dirpath).expanduser().absolute()

    for image_filename, feature_filename in tqdm.tqdm(
        zip(image_filenames, feature_filenames),
        desc="Visualizing images and features",
        total=len(image_filenames),
    ):
        if Path(image_filename).suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        feature_filepath = feature_dirpath / feature_filename
        with open(feature_filepath, "rb") as f:
            keypoints, _ = pickle.load(f)

        image = cv2.imread(str(image_dirpath / image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)

        if show_mode in ["show", "both"]:
            plt.figure(figsize=(20, 12))
            plt.suptitle(f"Image: {image_filename}")
            for keypoint in keypoints:
                color = np.random.random(3)
                circle = plt.Circle(
                    xy=keypoint.pt,
                    radius=keypoint.size,
                    color=color,
                    fill=False,
                )
                dxy = (np.cos(np.deg2rad(keypoint.angle)), np.sin(np.deg2rad(keypoint.angle)))
                x, y = keypoint.pt
                x2, y2 = x + dxy[0] * keypoint.size, y + dxy[1] * keypoint.size
                plt.plot([x, x2], [y, y2], color=color)
                plt.gca().add_patch(circle)
            plt.imshow(image, alpha=0.5)
            plt.show()

        if show_mode in ["dump", "both"]:
            dump_filepath = dump_dirpath / Path(image_filename).with_suffix(".png")
            overlay = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=4)
            cv2.imwrite(str(dump_filepath), overlay)


cli.add_command(extract)
cli.add_command(show)


if __name__ == "__main__":
    cli()
