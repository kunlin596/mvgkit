package(default_visibility = ["//visibility:public"])

load("@mvgkit_deps//:requirements.bzl", "requirement")

py_library(
    name = "frame",
    srcs = ["frame.py"],
    deps = [
        "//mvg:camera",
    ],
)

py_library(
    name = "reconstruction",
    srcs = ["reconstruction.py"],
    deps = [
        "//mvg:basic",
        requirement("plyfile"),
    ],
)

py_library(
    name = "visual_odometry",
    srcs = ["visual_odometry.py"],
    deps = [
        requirement("ipython"),
    ],
)
