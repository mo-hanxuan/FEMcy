"""Neo-Hookean material constitutive
"""

import taichi as ti
from mater_base import MaterBase


@ti.data_oriented
class NeoHookean(MaterBase):
    """elastic energy density ψ = C1 * (I1 - 3 - 2 * ln(J)) + D1 * (J - 1)**2,
       σ = J^(-1) * ∂ψ/∂F * F^T = 2*C1*J^(-1)*(B - I) + 2*D1*(J-1)*I
       https://en.wikipedia.org/wiki/Neo-Hookean_solid
    """

    def __init__(self, C1: float = 0.4, D1: float = 0.00025):
        self.type = "3d"
        self.C1 = C1
        self.D1 = D1
        self.dm = 3
        self.C = self.get_C()

    def get_C(self, ):
        """get ∂Δσ/∂Δε, the material Jacobian, can be utilized for both small and large deformation"""
        C1, D1 = self.C1, self.D1
        self.eye6 = eye6 = ti.Matrix([
            [1., 0., 0., 0., 0., 0.],  #
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]
        ])
        ### derivative of trace(epsilon) with respect to epsilon (Voigt notation)
        self.volumeStiffness = volumeStiffness = ti.Matrix([
            [1., 1., 1., 0., 0., 0.],  #
            [1., 1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]
        ])
        return 4. * C1 * eye6 + 2. * D1 * volumeStiffness  # ∂Δσ/∂Δε, the material Jacobian

    @ti.kernel
    def constitutiveOfSmallDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """update Cauchy stress and ddsdde for each integraion point"""
        C1, D1 = ti.static(self.C1, self.D1)
        eye = ti.Matrix([
            [1., 0., 0.],  #
            [0., 1., 0.],
            [0., 0., 1.]
        ])

        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]
            J = F.determinant()
            B = F @ F.transpose()  # left Cauchy-Green Strain tensor
            cauchy_stress[I] = \
                2. * C1 / J * (B - eye) + 2. * D1 * (J - 1.) * eye

        ### update material Jacobian if necessary
        # for I in ti.grouped(deformationGradient):
        #     ddsdde[I] = 4. * C1 * eye6 + 2. * D1 * volumeStiffness  # ∂Δσ/∂Δε, the material Jacobian

    @ti.kernel
    def constitutiveOfLargeDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """update Cauchy stress and ddsdde for each integraion point"""
        C1, D1 = ti.static(self.C1, self.D1)
        eye = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]
            J = F.determinant()
            B = F @ F.transpose()  # left Cauchy-Green Strain tensor
            cauchy_stress[I] = 2. * C1 / J * (B - eye) + 2. * D1 * (J - 1.) * eye

        ### update material Jacobian if necessary
        # for I in ti.grouped(deformationGradient):
        #     ddsdde[I] = 4. * C1 * self.eye6 + 2. * D1 * self.volumeStiffness  # ∂Δσ/∂Δε, the material Jacobian

    @ti.func
    def elasticEnergyDensity(self, deformationGradient):
        """elastic energy density ψ = C1 * (I1 - 3 - 2 * ln(J)) + D1 * (J - 1)**2 """
        F = deformationGradient
        J = F.determinant()
        B = F @ F.transpose()  # left Cauchy-Green Strain tensor
        return self.C1 * (B.trace() - 3. - 2. * ti.log(J)) + self.D1 * (J - 1.)**2
