import numpy as np
from scipy.spatial.transform import Rotation

from _mvgkit_camera_cppimpl import CameraMatrix
from mvgkit.common.utils import get_line_points_in_image
from pymvgkit_estimation import get_epilines, get_epipole


def plot_reprojection(
    image_L: np.ndarray,
    image_R: np.ndarray,
    points_L: np.ndarray,
    points_R: np.ndarray,
    points3d_L: np.ndarray,
    camera_matrix: CameraMatrix,
    R_RL: Rotation,
    t_RL: np.ndarray,
):
    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.imshow(image_L)
    plt.scatter(points_L[:, 0], points_L[:, 1], color="b", alpha=0.5, label="Input", s=20.0)
    reprojected_L = camera_matrix.project(points3d_L)
    plt.scatter(
        reprojected_L[:, 0], reprojected_L[:, 1], color="g", alpha=0.5, label="Reprojected", s=30.0
    )
    for i in range(len(points_L)):
        plt.plot(
            [points_L[i][0], reprojected_L[i][0]],
            [points_L[i][1], reprojected_L[i][1]],
            color="cyan",
            alpha=0.5,
        )
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(image_R)
    plt.scatter(points_R[:, 0], points_R[:, 1], color="b", alpha=0.5, label="Input", s=20.0)
    reprojected_R = camera_matrix.project(R_RL.apply(points3d_L) + t_RL)
    plt.scatter(
        reprojected_R[:, 0], reprojected_R[:, 1], color="g", alpha=0.5, label="Reprojected", s=30.0
    )
    for i in range(len(points_R)):
        plt.plot(
            [points_R[i][0], reprojected_R[i][0]],
            [points_R[i][1], reprojected_R[i][1]],
            color="cyan",
            alpha=0.5,
        )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_epipolar_lines(
    image_L: np.ndarray,
    image_R: np.ndarray,
    points_L: np.ndarray,
    points_R: np.ndarray,
    F_RL: np.ndarray,
):
    import matplotlib.pyplot as plt

    colors = np.random.random((len(points_L), 3)) * 0.7 + 0.1

    def _plot_line(ax, lines, width, height):
        for i, l in enumerate(lines):
            points = get_line_points_in_image(l, width, height)
            ax.plot(points[:, 0], points[:, 1], alpha=0.8, color=colors[i])

    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

    width = image_L.shape[1]
    height = image_L.shape[0]

    ax1.set_title("Corresponding epilines of (R) in (L)")
    ax1.imshow(image_L)
    ax1.set_xlim([0, width])
    ax1.set_ylim([height, 0])
    left_lines = get_epilines(x_R=points_R, F_RL=F_RL)
    left_epipole = get_epipole(F_RL)
    _plot_line(ax1, left_lines, width, height)
    ax1.scatter(points_L[:, 0], points_L[:, 1], color=colors)
    ax1.scatter([left_epipole[0]], [left_epipole[1]], color="r", s=30.0, alpha=1.0)

    width = image_R.shape[1]
    height = image_R.shape[0]

    ax2.set_title("Corresponding epilines of (L) in (R)")
    ax2.imshow(image_R)
    ax2.set_xlim([0, width])
    ax2.set_ylim([height, 0])
    right_lines = get_epilines(x_R=points_L, F_RL=F_RL.T)
    right_epipole = get_epipole(F_RL.T)
    _plot_line(ax2, right_lines, width, height)
    ax2.scatter(points_R[:, 0], points_R[:, 1], color=colors)
    ax2.scatter([right_epipole[0]], [right_epipole[1]], color="r", s=30.0, alpha=1.0)

    plt.tight_layout()
    plt.show()
