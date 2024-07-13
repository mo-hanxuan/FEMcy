"""the material constitutive with features of 
linear isotropic plane stress
"""

import taichi as ti
from mater_base import MaterBase


@ti.data_oriented
class LinearIsotropicPlaneStress(MaterBase):

    def __init__(self, modulus: float, poisson_ratio: float):
        self.type = "planeStress"  # only planeStress, planeStrain and 3d are available types
        self.modulus = modulus
        self.poisson_ratio = poisson_ratio
        self.dm = 2  # dimension for plane stress problem
        self.G = G = modulus / 2. / (1. + poisson_ratio)  # shear modulus
        c00 = modulus / (1. - poisson_ratio**2)
        c01 = c00 * poisson_ratio
        C = ti.Matrix([[c00, c01, 0.], [c01, c00, 0.], [0., 0., G]], ti.f64)

        C_6x6 = ti.Matrix(
            [  # voigt notation is related here, utilized to get 3D stress state for visulization
                [c00, c01, 0., 0., 0., 0.],  # sigma x
                [c01, c00, 0., 0., 0., 0.],  # sigma y
                [0., 0., 0., 0., 0., 0.],  # sigma z
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
        """linear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        nu = ti.static(self.poisson_ratio)
        eye_3d = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the deformation gradient at 3d
            F_3d = ti.Matrix.zero(ti.f64, 3, 3)
            F_3d[0:2, 0:2] = F[0:2, 0:2]
            F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

            ### get the infinitesimal strain E
            E = (F_3d + F_3d.transpose()) / 2. - eye_3d

            ### get the stress, voigt notation has been used here,
            ### modified later by different C at different gauss point
            stress_voigt = self.C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2], 2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            stress = ti.Matrix([[stress_voigt[0], stress_voigt[3], stress_voigt[4]],
                                [stress_voigt[3], stress_voigt[1], stress_voigt[5]],
                                [stress_voigt[4], stress_voigt[5], stress_voigt[2]]])
            ### get the cauchy stress
            cauchy_stress[I][0:2, 0:2] = stress[0:2, 0:2]

    @ti.kernel
    def constitutiveOfLargeDeform(self, deformationGradient: ti.template(), cauchy_stress: ti.template(),
                                  ddsdde: ti.template()):
        """geometric nonlinear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        nu = ti.static(self.poisson_ratio)
        eye_3d = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the deformation gradient at 3d
            F_3d = ti.Matrix.zero(ti.f64, 3, 3)
            F_3d[0:2, 0:2] = F[0:2, 0:2]
            F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

            ### get the Green Strain, E
            E = (F_3d.transpose() @ F_3d - eye_3d) / 2.

            ### get the PK2 stress, voigt notation has been used here,
            ### modified later by different C at different gauss point
            pk2_voigt = self.C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2], 2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            pk2 = ti.Matrix([
                [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],  #
                [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
            ])

            ### get the cauchy stress
            stress = F_3d @ pk2 @ F_3d.transpose() / F_3d.determinant()
            cauchy_stress[I][0:2, 0:2] = stress[0:2, 0:2]

    @ti.func
    def elasticEnergyDensity(self, deformationGradient):
        nu = ti.static(self.poisson_ratio)
        eye_3d = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        F = deformationGradient

        ### get the deformation gradient at 3d
        F_3d = ti.Matrix.zero(ti.f64, 3, 3)
        F_3d[0:2, 0:2] = F[0:2, 0:2]
        F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

        ### get the Green Strain, E
        E = (F_3d.transpose() @ F_3d - eye_3d) / 2.

        ### energy density
        E_voigt = ti.Vector([E[0, 0], E[1, 1], E[2, 2], 2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
        return E_voigt.dot(self.C_6x6 @ E_voigt) / 2.
