load("@mvgkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optical_flow",
    srcs = ["optical_flow.py"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python-headless"),
    ],
)
