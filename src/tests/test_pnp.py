import numpy as np
import pytest
from pymvgkit_estimation import EPnP
from scipy.spatial.transform import Rotation

from mvgkit.common.camera import CameraMatrix
from mvgkit.estimation.pnp import solve_epnp


@pytest.fixture
def points_W():
    points_W = np.array(
        [
            [-3.86612020e00, -2.68014320e00, 9.76254280e00],
            [-2.31005330e00, -3.17493940e00, 8.00051380e00],
            [-1.45376590e00, -5.45510500e-01, 8.20662250e00],
            [-3.90030430e00, -2.54002920e00, 1.00615239e01],
            [-2.72855890e00, -3.45708640e00, 7.42698100e00],
            [-9.57313700e-01, -3.67331090e00, 9.20382480e00],
            [-1.74139240e00, -1.02691800e00, 1.09270406e01],
            [-3.12460080e00, -2.18241630e00, 9.07461610e00],
            [-1.70463310e00, -1.77907580e00, 9.91418500e00],
            [-1.48450420e00, -8.15487500e-01, 9.33486330e00],
            [-1.90093420e00, 3.18392300e-01, 1.16547201e01],
            [8.61608600e-01, -1.99166100e00, 8.47978070e00],
            [-2.13787760e00, 5.54573100e-01, 1.19342702e01],
            [-9.31255100e-01, 6.54268900e-01, 9.39424100e00],
            [-1.86973120e00, -2.07429010e00, 9.93715460e00],
            [-1.44275010e00, 9.68396200e-01, 1.02638437e01],
            [7.61074700e-01, -7.72996200e-01, 9.23692570e00],
            [6.88115300e-01, 1.02402900e-01, 1.10545094e01],
            [-1.23726860e00, -1.60223010e00, 9.27543360e00],
            [-2.32308360e00, 9.13265600e-01, 8.45994610e00],
            [1.17797000e-02, -1.98766860e00, 1.07783384e01],
            [-3.33500200e-01, -2.12606010e00, 7.78667750e00],
            [1.46280400e-01, -1.26393770e00, 7.34125060e00],
            [8.05506000e-02, 8.04807200e-01, 1.09955195e01],
            [-2.52520840e00, -2.06492400e00, 7.92122940e00],
            [-5.88053800e-01, -5.07674100e-01, 7.88962560e00],
            [-2.32421000e00, -7.46599400e-01, 1.01166294e01],
            [-1.08728430e00, -1.05848410e00, 9.21903230e00],
            [-2.03277490e00, 7.04247300e-01, 1.06168022e01],
            [-7.57341300e-01, -2.61883630e00, 1.05300678e01],
            [-2.97395200e-01, 3.92943300e-01, 7.54239920e00],
            [-5.19056700e-01, -9.40556900e-01, 8.41633920e00],
            [2.67620700e-01, -1.79427640e00, 1.01524015e01],
            [6.08871300e-01, -2.71372580e00, 9.95720870e00],
            [6.04390800e-01, -2.23893420e00, 1.08922847e01],
            [-2.47437980e00, 1.73012100e-01, 9.38461060e00],
            [-1.58475420e00, 8.48802000e-01, 7.63801120e00],
            [-2.46812480e00, -1.23848230e00, 9.57952710e00],
            [-2.49092400e-01, -2.71257100e-01, 9.28377440e00],
            [-6.32290200e-01, -2.02859840e00, 9.66493800e00],
            [-1.10222240e00, 6.74006400e-01, 9.05788140e00],
            [-5.59823200e-01, -8.45050300e-01, 7.11732450e00],
            [8.56516000e-01, -1.57742960e00, 8.32304800e00],
            [8.91740000e-03, -1.96855830e00, 8.60932220e00],
            [-1.03387390e00, -2.36416750e00, 9.37038810e00],
            [-1.14158920e00, -1.83854730e00, 8.54340010e00],
            [-2.75697850e00, -4.23301500e-01, 8.39220210e00],
            [-2.11896730e00, 1.08573800e-01, 1.01537199e01],
            [4.60559800e-01, -1.14051870e00, 8.88246270e00],
            [-1.25566590e00, -2.77280890e00, 9.85386430e00],
            [-2.59072780e00, -8.75031300e-01, 9.52787070e00],
            [-5.32846500e-01, -2.43485440e00, 7.68282040e00],
            [-4.15522000e-01, -2.57833850e00, 9.10539070e00],
            [9.07526000e-01, 4.30579000e-01, 1.01368325e01],
            [-1.48315170e00, -1.60329490e00, 1.07726650e01],
            [8.87236300e-01, 2.55115900e-01, 7.93411770e00],
            [-1.56936360e00, 4.98137400e-01, 1.05108162e01],
            [-1.77161400e-01, -2.62082990e00, 9.61939000e00],
            [-1.02344160e00, 8.39729900e-01, 7.47887130e00],
            [8.59021100e-01, -1.41593600e00, 7.70606240e00],
            [-2.87114600e-01, -1.00666380e00, 9.83103120e00],
            [-1.75924390e00, -2.53951030e00, 1.03961767e01],
            [-1.07642350e00, 4.49677000e-02, 1.08178383e01],
            [-2.97103280e00, -4.75062000e-02, 1.02484172e01],
            [1.65799700e-01, -2.53065800e00, 7.64512230e00],
            [-6.15354000e-02, -2.64342160e00, 1.09002383e01],
            [8.72582300e-01, -1.21278520e00, 1.03983757e01],
            [3.83398500e-01, -2.38994660e00, 1.07775458e01],
            [-9.97211600e-01, -4.13388200e-01, 1.06172757e01],
            [-5.18340300e-01, -5.54367000e-01, 8.20133970e00],
            [1.87722100e-01, -1.84148160e00, 1.01946759e01],
            [-9.81246700e-01, -6.00725500e-01, 1.06551656e01],
            [-1.58507000e00, -2.67714900e00, 9.70013340e00],
            [-1.76723170e00, -2.64818180e00, 8.65262710e00],
            [-2.51881450e00, 5.17617900e-01, 9.12196920e00],
            [-1.87369220e00, -5.43917500e-01, 9.47854750e00],
            [-1.97345390e00, -6.71335200e-01, 7.26576230e00],
            [-2.57507830e00, -1.28793680e00, 7.87581570e00],
            [-2.79753250e00, 7.14851700e-01, 1.04624275e01],
            [8.19743200e-01, -8.03488600e-01, 8.90806040e00],
            [-1.97891710e00, -1.61576650e00, 1.00665788e01],
            [-2.78424120e00, 4.02986800e-01, 8.46585330e00],
            [8.70924400e-01, -2.18208320e00, 8.78870430e00],
            [-4.28942200e-01, -9.49314900e-01, 9.14052250e00],
            [-2.77631510e00, -4.68129400e-01, 8.65814040e00],
            [-6.54346000e-01, 6.58178400e-01, 7.11422280e00],
            [-2.17579850e00, -2.31527550e00, 9.44288760e00],
            [-1.91003620e00, -1.89035380e00, 7.15495090e00],
            [-1.03422050e00, -1.68788630e00, 1.08698025e01],
            [-1.57179300e00, -1.86814310e00, 9.06631390e00],
            [1.33626740e00, 1.52939700e-01, 1.04505474e01],
            [4.02846200e-01, 3.68698500e-01, 9.85353430e00],
            [1.86869950e00, 2.39622900e-01, 1.06714511e01],
            [-3.42596200e-01, -1.18931930e00, 8.72213620e00],
            [1.79792630e00, -9.65634400e-01, 7.25400690e00],
            [-5.43933300e-01, 1.38001960e00, 1.09121853e01],
            [-4.29710500e-01, -1.79577880e00, 7.59690970e00],
            [-1.98682290e00, -7.05815000e-01, 8.70655590e00],
            [-1.83187200e00, 1.25996450e00, 1.00186696e01],
            [-1.96206950e00, -1.31182850e00, 7.15052650e00],
        ]
    )

    return points_W


@pytest.fixture
def rotmat():
    return np.array(
        [
            [0.5, 0.0, 0.8660254],
            [-0.0, 1.0, 0.0],
            [-0.8660254, -0.0, 0.5],
        ]
    )


@pytest.fixture
def tvec():
    return np.array([-4.7631397, 0.0, 7.25])


@pytest.fixture
def camera_matrix():
    return CameraMatrix(520.0, 520.0, 325.0, 250.0, 0.0)


@pytest.fixture
def image_points(points_W, rotmat, tvec, camera_matrix):
    return camera_matrix.project(points_W @ rotmat.T + tvec)


def test_epnp_py(points_W, image_points, rotmat, tvec, camera_matrix):
    estimated_rotmat, estimated_tvec = solve_epnp(points_W, image_points, camera_matrix)

    points_C = points_W @ rotmat.T + tvec
    estimated_points_C = points_W @ estimated_rotmat.T + estimated_tvec
    np.testing.assert_allclose(estimated_points_C, points_C, atol=1e-6)


def test_epnp_cpp(points_W, image_points, rotmat, tvec, camera_matrix):
    estimated_pose = EPnP.solve(points_W, image_points, camera_matrix, True)
    R = Rotation.from_rotvec(estimated_pose[:3])
    points_C = R.apply(points_W) + tvec
    estimated_points_C = R.apply(points_W) + estimated_pose[3:]
    np.testing.assert_allclose(estimated_points_C, points_C, atol=1e-6)
