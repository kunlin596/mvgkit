package(default_visibility = ["//visibility:public"])

load("@mvgkit_deps//:requirements.bzl", "requirement")

py_library(
    name = "ui",
    srcs = ["ui.py"],
    deps = [
        requirement("pyvistaqt"),
        requirement("PySide6"),
        requirement("IPython"),
        "//mvg:sfm",
        "//mvg:streamer",
        "//mvg/models:kitti",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [
        ":ui",
        requirement("PySide6"),
        "//mvg:sfm",
    ],
)
