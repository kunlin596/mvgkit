py_binary(
    name = "feature_extractor",
    srcs = ["feature_extractor.py"],
    deps = [
        "//mvg:features",
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
