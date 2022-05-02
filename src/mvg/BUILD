load("@mvgkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "basic",
    srcs = ["basic.py"],
    deps = [
        requirement("numpy"),
        requirement("sympy"),
        requirement("scipy"),
    ],
)

py_library(
    name = "calibration",
    srcs = ["calibration.py"],
)

py_library(
    name = "camera",
    srcs = ["camera.py"],
    deps = ["//mvg:image_processing"],
)

py_library(
    name = "features",
    srcs = ["features.py"],
)

py_library(
    name = "homography",
    srcs = ["homography.py"],
    deps = ["//mvg:basic"],
)

py_library(
    name = "stereo",
    srcs = ["stereo.py"],
    deps = [
        "//mvg:basic",
        "//mvg:homography",
    ],
)

py_library(
    name = "image_processing",
    srcs = ["image_processing.py"],
    deps = [
        requirement("opencv-python-headless"),
    ],
)

py_library(
    name = "streamer",
    srcs = ["streamer.py"],
    deps = [
        "//mvg:features",
        "//mvg:image_processing",
        requirement("opencv-python-headless"),
    ],
)

py_library(
    name = "sfm",
    srcs = ["sfm.py"],
    deps = [
        "//mvg:basic",
        "//mvg:camera",
        "//mvg:features",
        "//mvg:stereo",
        "//mvg:streamer",
        "//mvg/mapping:frame",
        "//mvg/mapping:reconstruction",
        "//mvg/mapping:visual_odometry",
        requirement("opencv-python-headless"),
    ],
)
