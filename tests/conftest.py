import pytest
from pathlib import Path


def pytest_addoption(parser):
    default_path = Path(__file__).parent.absolute() / "data"
    parser.addoption("--data-root-path", action="store", default=default_path, type=str)
    parser.addoption("--intrinsics-rms-threshold", action="store", default=1.5, type=float)
    parser.addoption("--fundamental-rms-threshold", action="store", default=5.0, type=float)


@pytest.fixture()
def data_root_path(pytestconfig):
    return pytestconfig.getoption("data_root_path")


@pytest.fixture()
def intrinsics_rms_threshold(pytestconfig):
    return pytestconfig.getoption("intrinsics_rms_threshold")


@pytest.fixture()
def fundamental_rms_threshold(pytestconfig):
    return pytestconfig.getoption("fundamental_rms_threshold")
