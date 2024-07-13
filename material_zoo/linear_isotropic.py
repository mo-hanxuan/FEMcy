"""material constitutive with features of
linear isotropic
"""

import taichi as ti
from mater_base import MaterBase


@ti.data_oriented
class LinearIsotropic(MaterBase):  # linear isotropic material for 3d case

    def __init__(self, modulus: float, poisson_ratio: float):
        self.type = "3d"
        self.modulus = modulus
        self.poisson_ratio = nu = poisson_ratio
        self.dm = 3  # dimension
        self.G = G = modulus / 2. / (1. + nu)  # shear modulus
        c00 = modulus * (1. - nu) / (1. + nu) / (1. - 2. * nu)
        c01 = modulus * nu / (1. + nu) / (1. - 2. * nu)
        """the elastic tensor for linear isotropic material is refered to 
           https://help.solidworks.com/2010/English/SolidWorks/cworks/LegacyHelp/Simulation/Materials/Material_models/Linear_Elastic_Isotropic_Model.htm#:~:text=A%20material%20is%20said%20to,expansion%2C%20thermal%20conductivity%2C%20etc."""
        C_6x6 = ti.Matrix(
            [  # voigt notation is related here
                [c00, c01, c01, 0., 0., 0.],  # sigma x
                [c01, c00, c01, 0., 0., 0.],  # sigma y
                [c01, c01, c00, 0., 0., 0.],  # sigma z
                [0., 0., 0., G, 0., 0.],  # tau xy
                [0., 0., 0., 0., G, 0.],  # tau zx
                [0., 0., 0., 0., 0., G],  # tau yz
            ],
            ti.f64)

        self.C = C_6x6

    @ti.kernel
    def constitutiveOfSmallDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """constitutive use Green's strain and PK2 stress"""
        eye = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the infinitesimal strain, E
            E = (F + F.transpose()) / 2. - eye

            ### get the PK2 stress, voigt notation has been used here,
            ### modified later by different C at different gauss point
            s_voigt = ddsdde[I] @ ti.Vector([E[0, 0], E[1, 1], E[2, 2], 2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            cauchy_stress[I] = ti.Matrix([
                [s_voigt[0], s_voigt[3], s_voigt[4]],  #
                [s_voigt[3], s_voigt[1], s_voigt[5]],
                [s_voigt[4], s_voigt[5], s_voigt[2]]
            ])

    @ti.kernel
    def constitutiveOfLargeDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """constitutive use Green's strain and PK2 stress"""
        eye = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the Green Strain, E
            E = (F.transpose() @ F - eye) / 2.

            ### get the PK2 stress, voigt notation has been used here,
            ### modified later by different C at different gauss point
            pk2_voigt = ddsdde[I] @ ti.Vector([E[0, 0], E[1, 1], E[2, 2], 2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            pk2 = ti.Matrix([
                [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],  #
                [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
            ])

            ### get the cauchy stress
            cauchy_stress[I] = F @ pk2 @ F.transpose() / F.determinant()

    @ti.func
    def elasticEnergyDensity(self, deformationGradient):
        eye = ti.Matrix([
            [1., 0., 0.],  #
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        F = deformationGradient

        ### get the Green Strain, E
        E = (F.transpose() @ F - eye) / 2.

        ### elastic energy density
        E_voigt = ti.Vector([
            E[0, 0],
            E[1, 1],
            E[2, 2],  #
            2. * E[0, 1],
            2. * E[2, 0],
            2. * E[1, 2]
        ])
        return E_voigt.dot(self.C @ E_voigt) / 2.
