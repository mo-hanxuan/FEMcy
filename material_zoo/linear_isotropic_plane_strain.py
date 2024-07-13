"""the material constitutive with features of 
linear isotropic plane strain
"""

import taichi as ti
from mater_base import MaterBase


@ti.data_oriented
class LinearIsotropicPlaneStrain(MaterBase):

    def __init__(self, modulus: float, poisson_ratio: float):
        self.type = "planeStrain"
        self.modulus = modulus
        self.poisson_ratio = poisson_ratio
        self.dm = 2  # dimension for plane stress problem
        self.G = G = modulus / 2. / (1. + poisson_ratio)  # shear modulus
        term1 = modulus / (1. + poisson_ratio)
        term2 = poisson_ratio / (abs(1. - 2. * poisson_ratio) + 1.e-30)
        c00 = term1 * (1. + term2)
        c01 = term1 * term2
        C = ti.Matrix(
            [
                [c00, c01, 0.],  # sigma x
                [c01, c00, 0.],  # sigma y
                [0., 0., G],  # tau xy
            ],
            ti.f64)

        C_6x6 = ti.Matrix(
            [  # voigt notation is related here, utilized to get 3D stress state for visulization
                [c00, c01, c01, 0., 0., 0.],  # sigma x
                [c01, c00, c01, 0., 0., 0.],  # sigma y
                [c01, c01, 0., 0., 0., 0.],  # sigma z
                [0., 0., 0., G, 0., 0.],  # tau xy
                [0., 0., 0., 0., 0., 0.],  # tau zx
                [0., 0., 0., 0., 0., 0.],  # tau yz
            ],
            ti.f64)

        self.C = C
        self.C_6x6 = C_6x6

    @ti.kernel
    def constitutiveOfSmallDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """linear constitutive of plane strain, for small deformation
           get the stress of each integration point 
           according to deformation gradient"""
        eye = ti.Matrix([
            [1., 0.],
            [0., 1.],
        ])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the infinitesimal strain
            E = (F + F.transpose()) / 2. - eye

            ### get the stress
            E_voigt = ti.Vector([E[0, 0], E[1, 1], E[0, 1] + E[1, 0]])
            stress_voigt = ddsdde[I] @ E_voigt
            ### get the Cauchy stress
            cauchy_stress[I] = ti.Matrix([[stress_voigt[0], stress_voigt[2]], [stress_voigt[2], stress_voigt[1]]])

    @ti.kernel
    def constitutiveOfLargeDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """geometric nonlinear constitutive of plane strain,
           get the stress of each integration point 
           according to deformation gradient"""
        eye = ti.Matrix([
            [1., 0.],
            [0., 1.],
        ])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the Green Strain, E
            E = (F.transpose() @ F - eye) / 2.

            ### get the PK2 stress
            pk2_voigt = ddsdde[I] @ ti.Vector([E[0, 0], E[1, 1], E[0, 1] + E[1, 0]])
            pk2 = ti.Matrix([[pk2_voigt[0], pk2_voigt[2]], [pk2_voigt[2], pk2_voigt[1]]])
            ### get the Cauchy stress
            cauchy_stress[I] = F @ pk2 @ F.transpose() / F.determinant()

    @ti.func
    def elasticEnergyDensity(self, deformationGradient):
        eye = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        F = ti.Matrix.zero(ti.f64, 3, 3)
        F[0:2, 0:2] = deformationGradient[0:2, 0:2]
        F[2, 2] = 1.

        ### get the Green Strain, E
        E = (F.transpose() @ F - eye) / 2.

        ### elastic energy density
        E_voigt = ti.Vector([E[0, 0], E[1, 1], E[2, 2], 2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
        return E_voigt.dot(self.C_6x6 @ E_voigt) / 2.
