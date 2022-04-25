load("@mvgkit_deps//:requirements.bzl", "requirement")

py_library(
    name = "data_model",
    srcs = ["data_model.py"],
    data = glob(["data/**/*.jpg"]) + glob(["data/**/*.json"]),
    deps = [
        requirement("numpy"),
        "//mvg:camera",
        "//mvg:features",
        "//mvg:image_processing",
        "//mvg:stereo",
    ],
)

py_test(
    name = "test_basic",
    srcs = ["test_basic.py"],
    deps = [
        "//mvg:basic",
    ],
)

py_test(
    name = "test_camera",
    srcs = ["test_camera.py"],
    deps = [
        ":data_model",
        "//mvg:camera",
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "test_calibration",
    srcs = ["test_calibration.py"],
    deps = [
        ":data_model",
        "//mvg:basic",
        "//mvg:calibration",
        "//mvg:camera",
        "//mvg:homography",
        "//mvg:image_processing",
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "test_stereo",
    srcs = ["test_stereo.py"],
    deps = [
        ":data_model",
        "//mvg:basic",
        "//mvg:camera",
        "//mvg:features",
        "//mvg:homography",
        "//mvg:image_processing",
        "//mvg:stereo",
        requirement("opencv-python-headless"),
    ],
)
