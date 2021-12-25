from mvg.basic import SkewSymmetricMatrix3d
from numpy import allclose


def test_skew_symmetric_matrix():
    T1 = SkewSymmetricMatrix3d([1, 1, 1]).as_matrix()
    assert (T1.T == -T1).all()
    assert T1[0, 1] == -1
    assert T1[0, 2] == 1
    assert T1[1, 2] == -1

    T2 = SkewSymmetricMatrix3d.from_matrix(T1).as_vec()
    assert (T2 == [1, 1, 1]).all()

    assert allclose(
        SkewSymmetricMatrix3d.from_vec(T2).as_matrix(),
        SkewSymmetricMatrix3d.from_vec(-T2).as_matrix().T,
    )
