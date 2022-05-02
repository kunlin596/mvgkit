package(default_visibility = ["//visibility:public"])

load("@mvgkit_deps//:requirements.bzl", "requirement")

py_library(
    name = "sfm",
    srcs = ["sfm.py"],
    deps = [
        "//mvg:basic",
        "//mvg:camera",
        "//mvg:features",
        "//mvg:stereo",
        "//mvg:streamer",
        "//mvg/mapping/common:frame",
        "//mvg/mapping/common:reconstruction",
        "//mvg/mapping/common:visual_odometry",
        requirement("opencv-python-headless"),
    ],
)

py_library(
    name = "mapper_manager",
    srcs = ["mapper_manager.py"],
    deps = [
        "//mvg/mapping/mapper:sfm",
    ],
)
