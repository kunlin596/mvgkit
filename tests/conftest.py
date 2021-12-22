import pytest
from pathlib import Path


def pytest_addoption(parser):
    default_path = Path(__file__).parent.absolute() / "data/calibration/left"
    parser.addoption("--data-paths", action="store", default=[default_path], nargs="+")
    parser.addoption("--intrinsics-rms-threshold", action="store", default=1.5, type=float)
    parser.addoption("--fundamental-rms-threshold", action="store", default=0.4, type=float)


@pytest.fixture()
def data_paths(pytestconfig):
    return pytestconfig.getoption("data_paths")


@pytest.fixture()
def intrinsics_rms_threshold(pytestconfig):
    return pytestconfig.getoption("intrinsics_rms_threshold")


@pytest.fixture()
def fundamental_rms_threshold(pytestconfig):
    return pytestconfig.getoption("fundamental_rms_threshold")
