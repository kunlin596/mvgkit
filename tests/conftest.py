import pytest
from pathlib import Path


def pytest_addoption(parser):
    default_path = Path(__file__).parent.absolute() / "data/calibration/left"
    parser.addoption("--data-paths", action="store", default=[default_path], nargs="+")
    parser.addoption("--rms-threshold", action="store", default=7.0, type=float)


@pytest.fixture()
def data_paths(pytestconfig):
    return pytestconfig.getoption("data_paths")


@pytest.fixture()
def rms_threshold(pytestconfig):
    return pytestconfig.getoption("rms_threshold")
