import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mvgkit.common import get_rigid_body_motion


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def rotation(rng):
    return Rotation.from_rotvec(rng.random(3))


@pytest.fixture
def translation(rng):
    return rng.random(3)


@pytest.fixture
def points(rng):
    return rng.random((100, 3))


@pytest.fixture
def transformed_points(points, rotation, translation):
    return rotation.apply(points) + translation


def test_rigid_body_motion(rotation, translation, points, transformed_points):
    pose = get_rigid_body_motion(points, transformed_points)
    np.testing.assert_allclose(pose, np.r_[rotation.as_rotvec(), translation])
