load("@mvgkit_deps//:requirements.bzl", "requirement")

py_binary(
    name = "feature_extractor",
    srcs = ["feature_extractor.py"],
    deps = [
        "//mvg:features",
    ],
)

py_binary(
    name = "naive_sfm",
    srcs = ["naive_sfm.py"],
    deps = [
        "//mvg/mapping/mapper:sfm",
        "//mvg/models:kitti",
    ],
)

py_binary(
    name = "kitti_downloader",
    srcs = ["kitti_downloader.py"],
)

py_binary(
    name = "kitti_simple_visualizer",
    srcs = ["kitti_simple_visualizer.py"],
    deps = [
        "//mvg:features",
        "//mvg:homography",
        "//mvg:image_processing",
        "//mvg:stereo",
        "//mvg/models:kitti",
    ],
)

py_binary(
    name = "run_optical_flow",
    srcs = ["run_optical_flow.py"],
    deps = [
        "//mvg:features",
        "//mvg/algorithms:optical_flow",
        requirement("ipython"),
        requirement("matplotlib"),
        requirement("opencv-python-headless"),
    ],
)
