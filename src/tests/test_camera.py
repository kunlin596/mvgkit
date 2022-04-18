#!/usr/bin/env python3
"""
FIXME: The tests below are incomplete.
"""

import numpy as np
from mvg.camera import CameraMatrix, RadialDistortionModel, TangentialDistortionModel
from pytest import fixture


@fixture()
def camera_matrix():
    return CameraMatrix(
        fx=535.35887264164,
        fy=535.6467965086524,
        cx=342.63131029730295,
        cy=233.77551538388735,
        s=0.5369513302298261,
    )


@fixture
def radial_distortion():
    return RadialDistortionModel(-0.4, 0.5, -0.57)


@fixture
def tangential_distortion():
    return TangentialDistortionModel(0.01, 0.05)


@fixture
def rng():
    return np.random.default_rng(42)


@fixture
def num_points():
    return 10


@fixture
def points_C(num_points):
    points_C = np.dstack(
        np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
    )
    points_C = points_C.reshape(-1, 2)
    points_C = np.hstack([points_C, np.ones((len(points_C), 1)) * 2.0])
    return points_C
