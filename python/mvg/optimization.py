#!/usr/bin/env python3

import math
from mvg import basic
import sympy as sp


class ProjectionFunctional:
    def _init_rotation_mat(self):
        # NOTE: Rodrigues's rotation formula, check more
        self.r1, self.r2, self.r3 = sp.symbols("r1 r2 r3", real=True)
        self.rmat = basic.get_symbolic_rodrigues_rotmat(r1=self.r1, r2=self.r2, r3=self.r3)

    def _init_translation_vec(self):
        self.t1, self.t2, self.t3 = sp.symbols("t1 t2 t3", real=True)
        self.tvec = sp.Matrix([self.t1, self.t2, self.t3])

    def _init_camera_matrix(self):
        self.fx = sp.Symbol("fx", real=True)
        self.fy = sp.Symbol("fy", real=True)
        self.cx = sp.Symbol("cx", real=True)
        self.cy = sp.Symbol("cy", real=True)
        self.s = sp.Symbol("s", real=True)
        self.K = sp.Matrix([[self.fx, self.s, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def __init__(self):
        # Parameters
        self._init_rotation_mat()
        self._init_camera_matrix()
        self._init_translation_vec()

        # Inputs
        self.X1, self.X2, self.X3 = sp.symbols("X1 X2 X3", real=True)
        self.X = sp.Matrix([self.X1, self.X2, self.X3, 1.0])

        self.x1, self.x2 = sp.symbols("x1 x2", real=True)
        self.x = sp.Matrix([self.x1, self.x2, 1.0])

        # Functional
        self.T = self.rmat.row_join(self.tvec)
        self.x_hat = self.K @ self.T @ self.X
        self.x_hat /= self.x_hat[-1]
        self.residual = self.x - self.x_hat
        self.functional = self.residual.T @ self.residual

        self.jac = self.functional.jacobian(
            sp.Matrix(
                [
                    self.r1,
                    self.r2,
                    self.r3,
                    self.t1,
                    self.t2,
                    self.t3,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    self.s,
                ]
            )
        )

    def _create_subs(self, image_point, object_point, camera_matrix, camera_pose):
        rvec = camera_pose.R.as_rotvec()
        tvec = camera_pose.t
        return [
            #
            (self.x1, image_point[0]),
            (self.x2, image_point[1]),
            #
            (self.X1, object_point[0]),
            (self.X2, object_point[1]),
            (self.X3, object_point[2]),
            #
            (self.r1, rvec[0]),
            (self.r2, rvec[1]),
            (self.r3, rvec[2]),
            #
            (self.t1, tvec[0]),
            (self.t2, tvec[1]),
            (self.t3, tvec[2]),
            #
            (self.fx, camera_matrix.fx),
            (self.fy, camera_matrix.fy),
            (self.cx, camera_matrix.cx),
            (self.cy, camera_matrix.cy),
            (self.s, camera_matrix.s),
        ]

    def compute_jacobian_value(self, image_point, object_point, camera_matrix, camera_pose):
        subs = self._create_subs(image_point, object_point, camera_matrix, camera_pose)
        return self.jac.subs(subs)

    def compute_reprojection_error(self, image_point, object_point, camera_matrix, camera_pose):
        subs = self._create_subs(image_point, object_point, camera_matrix, camera_pose)
        return math.sqrt(float(self.functional.subs(subs)[0]))


if __name__ == "__main__":
    p = ProjectionFunctional()
