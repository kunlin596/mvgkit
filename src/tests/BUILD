py_test(
    name = "test_basic",
    srcs = ["test_basic.py"],
    main = "test_basic.py",
    deps = ["//mvg:basic"],
)

py_test(
    name = "test_camera",
    srcs = ["test_camera.py"],
    main = "test_camera.py",
    deps = [
        "//mvg:basic",
        "//mvg:camera",
    ],
)

py_test(
    name = "test_calibration",
    srcs = ["test_calibration.py"],
    main = "test_calibration.py",
    deps = [
        "//mvg:basic",
        "//mvg:calibration",
        "//mvg:camera",
        "//mvg:homography",
        "//mvg:image_processing",
    ],
)

py_test(
    name = "test_stereo",
    srcs = ["test_stereo.py"],
    main = "test_stereo.py",
    deps = [
        "//mvg:basic",
        "//mvg:camera",
        "//mvg:features",
        "//mvg:homography",
        "//mvg:image_processing",
        "//mvg:stereo",
    ],
)
