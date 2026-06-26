"""Guard against import regressions across the mvgkit package.

The refactor in #88 moved several modules but left stale intra-package import
paths, which silently broke a number of submodules. This test imports every
discoverable submodule so that such breakage fails loudly.

A small set of modules is intentionally skipped: they depend on optional
third-party packages or on the work-in-progress mapping / structure-from-motion
pipeline (which still targets a removed estimation API). Each carries the reason
it is skipped.
"""

import importlib
import pkgutil

import pytest

import mvgkit

# Module -> reason it is not expected to import in the core test environment.
WIP_OR_OPTIONAL_MODULES = {
    "mvgkit.apps.stereo_viewer": "imports PySide2 (should be PySide6); Qt app is WIP",
    "mvgkit.scripts.kitti_simple_visualizer": "requires optional dependency pyvista",
    "mvgkit.mapping.reconstruction": "requires optional dependency plyfile",
    "mvgkit.mapping.mapper_manager": "WIP mapping/SfM pipeline (needs plyfile + API migration)",
    "mvgkit.mapping.sfm": "WIP mapping/SfM pipeline (needs plyfile + API migration)",
    "mvgkit.mapping.visual_odometry": "WIP VO pipeline targets the removed essential/triangulation API",
    "mvgkit.scripts.naive_sfm": "depends on the WIP mapping/SfM pipeline",
}


def _discover_submodules():
    names = []
    for module in pkgutil.walk_packages(mvgkit.__path__, mvgkit.__name__ + "."):
        names.append(module.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _discover_submodules())
def test_submodule_imports(module_name):
    if module_name in WIP_OR_OPTIONAL_MODULES:
        pytest.skip(WIP_OR_OPTIONAL_MODULES[module_name])
    importlib.import_module(module_name)


def test_compiled_extensions_import():
    """The pybind11 extension modules must load (RPATH/co-location sanity)."""
    importlib.import_module("pymvgkit_common")
    importlib.import_module("pymvgkit_estimation")
