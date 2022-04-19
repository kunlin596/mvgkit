py_binary(
    name = "kitti_downloader",
    srcs = ["kitti_downloader.py"],
)

py_binary(
    name = "kitti_simple_visualizer",
    srcs = ["kitti_simple_visualizer.py"],
    deps = [
        "//mvg:image_processing",
        "//mvg/models:kitti",
    ],
)
