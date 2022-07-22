import os
from pathlib import Path

from pytest import fixture

from .data_fixtures import (  # noqa: F401
    aloe_stereo_data_pack,
    book_stereo_data_pack,
    leuven_stereo_data_pack,
)


@fixture
def intrinsics_rms_threshold():
    return 2.6


@fixture
def fundamental_rms_threshold():
    return 1.5


@fixture
def stereo_reprojection_rms_threshold():
    return 0.72


@fixture
def test_root_path():
    return Path(os.path.dirname(__file__)).parent / "tests"


@fixture
def data_root_path(test_root_path):
    return test_root_path / "data"
