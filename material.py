import numpy as np
import taichi as ti


@ti.data_oriented
class Linear_isotropic_planeStress:
    def __init__(self, modulus: float, poisson_ratio: float):
        self.type = "planeStress"  # only planeStress, planeStrain and 3d are available types
        self.modulus = modulus; self.poisson_ratio = poisson_ratio
        self.dm = 2  # dimension for plane stress problem
        self.G = G = modulus / 2. / (1. + poisson_ratio)  # shear modulus
        c00 = modulus / (1. - poisson_ratio**2)
        c01 = c00 * poisson_ratio
        C = ti.Matrix([
            [c00, c01, 0.],
            [c01, c00, 0.], 
            [0.,  0.,  G ]
        ], ti.f64)

        C_6x6 = ti.Matrix([  # voigt notation is related here, utilized to get 3D stress state for visulization
            [c00, c01, 0., 0., 0., 0.],  # sigma x
            [c01, c00, 0., 0., 0., 0.],  # sigma y
            [0., 0., 0., 0., 0., 0.],  # sigma z
            [0.,  0.,  0., G, 0., 0. ],  # tau xy
            [0.,  0.,  0., 0., 0., 0.],  # tau zx
            [0.,  0.,  0., 0., 0., 0.],  # tau yz
        ], ti.f64)
        
        self.C = C
        self.C_6x6 = C_6x6


    @ti.kernel
    def constitutive_smallDeform(self, deformationGradient: ti.template(), 
                            cauchy_stress: ti.template(), ddsdde: ti.template()):
        """linear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        nu = ti.static(self.poisson_ratio)
        eye_3d = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the deformation gradient at 3d
            F_3d = ti.Matrix([[0. for _ in range(3)] for _ in range(3)])
            F_3d[0:2, 0:2] = F[0:2, 0:2]
            F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

            ### get the infinitesimal strain E
            E = (F_3d + F_3d.transpose()) / 2. - eye_3d

            ### get the stress, voigt notation has been used here, 
            ### modified later by different C at different gauss point
            stress_voigt = self.C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                    2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            stress = ti.Matrix([ 
                [stress_voigt[0], stress_voigt[3], stress_voigt[4]],
                [stress_voigt[3], stress_voigt[1], stress_voigt[5]],
                [stress_voigt[4], stress_voigt[5], stress_voigt[2]]
            ])
            ### get the cauchy stress
            cauchy_stress[I][0:2, 0:2] = stress[0:2, 0:2]
    

    @ti.kernel
    def constitutive_largeDeform(self, deformationGradient: ti.template(), 
                                 cauchy_stress: ti.template(), ddsdde: ti.template()):
        """geometric nonlinear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        nu = ti.static(self.poisson_ratio)
        eye_3d = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the deformation gradient at 3d
            F_3d = ti.Matrix([[0. for _ in range(3)] for _ in range(3)])
            F_3d[0:2, 0:2] = F[0:2, 0:2]
            F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

            ### get the Green Strain, E
            E = (F_3d.transpose() @ F_3d - eye_3d) / 2.

            ### get the PK2 stress, voigt notation has been used here, 
            ### modified later by different C at different gauss point
            pk2_voigt = self.C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            pk2 = ti.Matrix([ 
                [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],
                [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
            ])

            ### get the cauchy stress
            stress = F_3d @ pk2 @ F_3d.transpose() / F_3d.determinant()
            cauchy_stress[I][0:2, 0:2] = stress[0:2, 0:2]
    

@ti.data_oriented
class Linear_isotropic_planeStrain:
    def __init__(self, modulus: float, poisson_ratio: float):
        self.type = "planeStrain"
        self.modulus = modulus; self.poisson_ratio = poisson_ratio
        self.dm = 2  # dimension for plane stress problem
        self.G = G = modulus / 2. / (1. + poisson_ratio)  # shear modulus
        term1 = modulus / (1. + poisson_ratio)
        term2 = poisson_ratio / (abs(1. - 2. * poisson_ratio) + 1.e-30)
        c00 = term1 * (1. + term2)
        c01 = term1 * term2
        C = ti.Matrix([
            [c00, c01, 0.],  # sigma x
            [c01, c00, 0.],  # sigma y
            [0.,  0.,  G ],  # tau xy
        ], ti.f64)
        
        C_6x6 = ti.Matrix([  # voigt notation is related here, utilized to get 3D stress state for visulization
            [c00, c01, c01, 0., 0., 0.],  # sigma x
            [c01, c00, c01, 0., 0., 0.],  # sigma y
            [c01, c01, 0., 0., 0., 0.],  # sigma z
            [0.,  0.,  0., G, 0., 0. ],  # tau xy
            [0.,  0.,  0., 0., 0., 0.],  # tau zx
            [0.,  0.,  0., 0., 0., 0.],  # tau yz
        ], ti.f64)

        self.C = C
        self.C_6x6 = C_6x6


    @ti.kernel
    def constitutive_smallDeform(self, deformationGradient: ti.template(), 
                            cauchy_stress: ti.template(), ddsdde: ti.template()):
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
            E_voigt = ti.Vector([E[0, 0], E[1, 1], 
                                E[0, 1] + E[1, 0]])
            stress_voigt = ddsdde[I] @ E_voigt
            ### get the Cauchy stress
            cauchy_stress[I] = ti.Matrix([[stress_voigt[0], stress_voigt[2]], 
                                          [stress_voigt[2], stress_voigt[1]]])


    @ti.kernel
    def constitutive_largeDeform(self, deformationGradient: ti.template(), 
                                 cauchy_stress: ti.template(), ddsdde: ti.template()):
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
            pk2_voigt = ddsdde[I] @ ti.Vector([E[0, 0], E[1, 1], 
                                               E[0, 1] + E[1, 0]])
            pk2 = ti.Matrix([ 
                [pk2_voigt[0], pk2_voigt[2]],
                [pk2_voigt[2], pk2_voigt[1]]
            ])
            ### get the Cauchy stress  
            cauchy_stress[I] = F @ pk2 @ F.transpose() / F.determinant()


@ti.data_oriented
class Linear_isotropic:  # linear isotropic material for 3d case
    def __init__(self, modulus: float, poisson_ratio: float):
        self.type = "3d"
        self.modulus = modulus; self.poisson_ratio = nu = poisson_ratio
        self.dm = 3  # dimension
        self.G = G = modulus / 2. / (1. + nu)  # shear modulus
        c00 = modulus * (1. - nu) / (1. + nu) / (1. - 2. * nu)
        c01 = modulus * nu / (1. + nu) / (1. - 2. * nu)

        """the elastic tensor for linear isotropic material is refered to 
           https://help.solidworks.com/2010/English/SolidWorks/cworks/LegacyHelp/Simulation/Materials/Material_models/Linear_Elastic_Isotropic_Model.htm#:~:text=A%20material%20is%20said%20to,expansion%2C%20thermal%20conductivity%2C%20etc."""
        C_6x6 = ti.Matrix([  # voigt notation is related here
            [c00, c01, c01, 0., 0., 0.],  # sigma x
            [c01, c00, c01, 0., 0., 0.],  # sigma y
            [c01, c01, c00, 0., 0., 0.],  # sigma z
            [0.,  0.,  0.,  G , 0., 0.],  # tau xy
            [0.,  0.,  0.,  0., G , 0.],  # tau zx
            [0.,  0.,  0.,  0., 0., G ],  # tau yz
        ], ti.f64)

        self.C = C_6x6


    @ti.kernel 
    def constitutive_smallDeform(self, deformationGradient: ti.template(), 
                                 cauchy_stress: ti.template(), ddsdde: ti.template()):
        """constitutive use Green's strain and PK2 stress"""
        eye = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the infinitesimal strain, E
            E = (F + F.transpose()) / 2. - eye

            ### get the PK2 stress, voigt notation has been used here, 
            ### modified later by different C at different gauss point
            s_voigt = ddsdde[I] @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                          2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            cauchy_stress[I] = ti.Matrix([ 
                [s_voigt[0], s_voigt[3], s_voigt[4]],
                [s_voigt[3], s_voigt[1], s_voigt[5]],
                [s_voigt[4], s_voigt[5], s_voigt[2]]
            ])


    @ti.kernel 
    def constitutive_largeDeform(self, deformationGradient: ti.template(), 
                                 cauchy_stress: ti.template(), ddsdde: ti.template()):
        """constitutive use Green's strain and PK2 stress"""
        eye = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for I in ti.grouped(deformationGradient):
            F = deformationGradient[I]

            ### get the Green Strain, E
            E = (F.transpose() @ F - eye) / 2.

            ### get the PK2 stress, voigt notation has been used here, 
            ### modified later by different C at different gauss point
            pk2_voigt = ddsdde[I] @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                               2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
            pk2 = ti.Matrix([ 
                [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],
                [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
            ])

            ### get the cauchy stress
            cauchy_stress[I] = F @ pk2 @ F.transpose() / F.determinant()


@ti.data_oriented
class NeoHookean(object):
    def __init__(self, C1: float=0.4, D1: float=0.00025):
        self.type = "3d"
        self.C1 = C1
        self.D1 = D1
        self.dm = 3
        self.C = self.get_C()


    def get_C(self, ):
        """get ∂Δσ/∂Δε, the material Jacobian, can be utilized for both small and large deformation"""
        C1, D1 = self.C1, self.D1
        self.eye6 = eye6 = ti.Matrix([[1., 0., 0., 0., 0., 0.], 
                                      [0., 1., 0., 0., 0., 0.], 
                                      [0., 0., 1., 0., 0., 0.], 
                                      [0., 0., 0., 1., 0., 0.], 
                                      [0., 0., 0., 0., 1., 0.], 
                                      [0., 0., 0., 0., 0., 1.]])
        ### derivative of trace(epsilon) with respect to epsilon (Voigt notation)
        self.volumeStiffness = volumeStiffness = ti.Matrix([[1., 1., 1., 0., 0., 0.], 
                                                            [1., 1., 1., 0., 0., 0.], 
                                                            [1., 1., 1., 0., 0., 0.], 
                                                            [0., 0., 0., 0., 0., 0.], 
                                                            [0., 0., 0., 0., 0., 0.], 
                                                            [0., 0., 0., 0., 0., 0.]])
        return 4. * C1 * eye6 + 2. * D1 * volumeStiffness  # ∂Δσ/∂Δε, the material Jacobian


    @ti.kernel
    def constitutive_smallDeform(self, deformationGradient: ti.template(), 
                     cauchy_stress: ti.template(), ddsdde: ti.template()):
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
        #     ddsdde[I] = 4. * C1 * eye6 + 2. * D1 * volumeStiffness  # ∂Δσ/∂Δε, the material Jacobian


    @ti.kernel
    def constitutive_largeDeform(self, deformationGradient: ti.template(), 
                     cauchy_stress: ti.template(), ddsdde: ti.template()):
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