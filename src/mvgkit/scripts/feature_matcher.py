#!/usr/bin/env python3

from __future__ import annotations

import glob
import itertools
import os
import pickle
import re
from pathlib import Path

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from mvgkit import features
from mvgkit.common.utils import cv2keypoints_to_nparray


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--feature-dirpath",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="directory path of features",
)
@click.option(
    "--output-dirpath",
    "-o",
    type=click.Path(),
    default=str(Path.cwd().absolute()),
    help="directory path of output",
)
@click.option(
    "--match-mode",
    "-m",
    type=click.Choice(["sequential", "exhaustive"]),
    default="sequential",
    help="matching mode",
)
def match(feature_dirpath: str, output_dirpath: str, match_mode: str):
    feature_dirpath = Path(feature_dirpath)
    output_dirpath = Path(output_dirpath)

    feature_filenames = sorted(glob.glob("*.feat", root_dir=feature_dirpath))
    all_keypoints = []
    all_descriptors = []
    for feature_filename in tqdm.tqdm(
        feature_filenames,
        desc="Loading features",
        total=len(feature_filenames),
    ):
        feature_filepath = feature_dirpath / feature_filename
        with open(feature_filepath, "rb") as f:
            keypoints, descriptors = pickle.load(f)
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)

    matcher = features.Matcher()

    os.makedirs(output_dirpath, exist_ok=True)
    if match_mode == "sequential":
        for i in tqdm.trange(len(all_descriptors) - 1, desc=f"Matching features {match_mode}"):
            filename1 = feature_filenames[i].split(".")[0]
            filename2 = feature_filenames[i + 1].split(".")[0]
            query_indices, train_indices = matcher.match(all_descriptors[i], all_descriptors[i + 1])
            output_filename = f"{filename1}_{filename2}.match"
            output_filepath = output_dirpath / output_filename
            with open(output_filepath, "wb") as f:
                pickle.dump((query_indices, train_indices), f)
    elif match_mode == "exhaustive":
        for i, j in tqdm.tqdm(
            itertools.combinations(range(len(all_descriptors)), 2),
            desc=f"Matching features {match_mode}",
            total=len(all_descriptors) * (len(all_descriptors) - 1) // 2,
        ):
            filename1 = feature_filenames[i].split(".")[0]
            filename2 = feature_filenames[j].split(".")[0]
            query_indices, train_indices = matcher.match(all_descriptors[i], all_descriptors[j])
            output_filename = f"{filename1}_{filename2}.match"
            output_filepath = output_dirpath / output_filename
            with open(output_filepath, "wb") as f:
                pickle.dump((query_indices, train_indices), f)
    else:
        raise RuntimeError(f"Unknown matching mode: {match_mode}.")


@click.command()
@click.option(
    "--image-dirpath",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="directory path of images",
)
@click.option(
    "--feature-dirpath",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="directory path of features",
)
@click.option(
    "--match-dirpath",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="directory path of matches",
)
@click.option(
    "--show-mode",
    "-s",
    type=click.Choice(["show", "dump", "both"]),
    default="show",
    help="showing mode",
)
@click.option(
    "--output-dirpath",
    "-o",
    type=click.Path(),
    default=str(Path.cwd().absolute()),
    help="directory path of output",
)
def show(
    image_dirpath: str,
    feature_dirpath: str,
    match_dirpath: str,
    show_mode: str,
    output_dirpath: str,
):
    image_dirpath = Path(image_dirpath)
    feature_dirpath = Path(feature_dirpath)
    match_dirpath = Path(match_dirpath)

    match_filenames = sorted(glob.glob("*.match", root_dir=match_dirpath))

    image_filename_pattern = re.compile(r"^(.*\.(jpeg|jpg|png))$", re.IGNORECASE)
    image_filenames = sorted(
        [
            filename
            for filename in os.listdir(image_dirpath)
            if image_filename_pattern.match(filename)
        ]
    )

    feature_filenames = sorted(glob.glob("*.feat", root_dir=feature_dirpath))
    assert len(feature_filenames) == len(
        image_filenames
    ), "Number of features and images must be the same. Check your input data."

    image_file_extension = image_filenames[0].split(".")[-1]

    if show_mode in ["dump", "both"]:
        os.makedirs(output_dirpath, exist_ok=True)

    for match_filename in tqdm.tqdm(
        match_filenames,
        total=len(match_filenames),
        desc="Showing feature matches",
    ):
        image_filename1, image_filename2 = match_filename.split(".")[0].split("_")

        image1 = cv2.imread(str(image_dirpath / f"{image_filename1}.{image_file_extension}"))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(str(image_dirpath / f"{image_filename2}.{image_file_extension}"))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        feature_filepath1 = feature_dirpath / f"{image_filename1}.feat"
        feature_filepath2 = feature_dirpath / f"{image_filename2}.feat"

        with open(feature_filepath1, "rb") as f:
            keypoints1, _ = pickle.load(f)
            keypoints1 = cv2keypoints_to_nparray(keypoints1)

        with open(feature_filepath2, "rb") as f:
            keypoints2, _ = pickle.load(f)
            keypoints2 = cv2keypoints_to_nparray(keypoints2)

        match_image = np.hstack([image1, image2])

        plt.figure(0, figsize=(20, 10))
        plt.suptitle(f"Feature matches: {image_filename1} - {image_filename2}")
        plt.imshow(match_image)
        plt.scatter(
            keypoints1[:, 0],
            keypoints1[:, 1],
            c="r",
            s=3,
            label=f"{image_filename1}",
        )
        plt.scatter(
            keypoints2[:, 0] + image1.shape[1],
            keypoints2[:, 1],
            c="r",
            s=1,
            label=f"{image_filename2}",
        )
        for kp1, kp2 in zip(keypoints1, keypoints2):
            plt.plot(
                [kp1[0], kp2[0] + image1.shape[1]],
                [kp1[1], kp2[1]],
                c="b",
                linewidth=0.5,
            )
        plt.tight_layout()

        if show_mode in ["show", "both"]:
            plt.show()
        elif show_mode in ["dump", "both"]:
            output_filename = f"{image_filename1}_{image_filename2}.png"
            output_filepath = output_dirpath / output_filename
            plt.savefig(output_filepath)


cli.add_command(match)
cli.add_command(show)
