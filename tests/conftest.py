import pytest
from pathlib import Path

from stereo_data_fixtures import book_stereo_data_pack, leuven_stereo_data_pack  # noqa: F401


def pytest_addoption(parser):
    default_path = Path(__file__).parent.absolute() / "data"
    parser.addoption("--data-root-path", action="store", default=default_path, type=str)
    parser.addoption("--intrinsics-rms-threshold", action="store", default=1.5, type=float)
    parser.addoption("--fundamental-rms-threshold", action="store", default=1.5, type=float)
    parser.addoption("--stereo-reprojection-rms-threshold", action="store", default=0.5, type=float)


@pytest.fixture(scope="package")
def data_root_path(pytestconfig):
    return pytestconfig.getoption("data_root_path")


@pytest.fixture(scope="package")
def intrinsics_rms_threshold(pytestconfig):
    return pytestconfig.getoption("intrinsics_rms_threshold")


@pytest.fixture(scope="package")
def fundamental_rms_threshold(pytestconfig):
    return pytestconfig.getoption("fundamental_rms_threshold")


@pytest.fixture(scope="package")
def stereo_reprojection_rms_threshold(pytestconfig):
    return pytestconfig.getoption("stereo_reprojection_rms_threshold")
