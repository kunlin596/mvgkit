package(default_visibility = ["//visibility:public"])

load("@mvgkit_deps//:requirements.bzl", "requirement")

py_library(
    name = "kitti",
    srcs = ["kitti.py"],
    deps = [
        "//mvg:basic",
        "//mvg:camera",
        "//mvg:image_processing",
        requirement("tqdm"),
    ],
)
