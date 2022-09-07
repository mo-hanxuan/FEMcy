import numpy as np
import taichi as ti


class Linear_isotropic_planeStress:
    def __init__(self, modulus: float, poisson_ratio: float):
        self.modulus = modulus; self.poisson_ratio = poisson_ratio
        self.dm = 2  # dimension for plane stress problem
        self.G = G = modulus / 2. / (1. + poisson_ratio)  # shear modulus
        c00 = modulus / (1. - poisson_ratio**2)
        c01 = c00 * poisson_ratio
        C = np.array([    # elastic constants in Voigt notation, len(C) = dm + dm*(dm-1)/2, i.e. 2d -> 3, 3d -> 6
            [c00, c01, 0.],
            [c01, c00, 0.], 
            [0.,  0.,  G ]
        ])
        ti_C = ti.Matrix([
            [c00, c01, 0.],
            [c01, c00, 0.], 
            [0.,  0.,  G ]
        ], ti.f64)

        C_6x6 = np.array([  # voigt notation is related here, utilized to get 3D stress state for visulization
            [c00, c01, 0., 0., 0., 0.],  # sigma x
            [c01, c00, 0., 0., 0., 0.],  # sigma y
            [0., 0., 0., 0., 0., 0.],  # sigma z
            [0.,  0.,  0., G, 0., 0. ],  # tau xy
            [0.,  0.,  0., 0., 0., 0.],  # tau zx
            [0.,  0.,  0., 0., 0., 0.],  # tau yz
        ])
        ti_C_6x6 = ti.Matrix([  # voigt notation is related here, utilized to get 3D stress state for visulization
            [c00, c01, 0., 0., 0., 0.],  # sigma x
            [c01, c00, 0., 0., 0., 0.],  # sigma y
            [0., 0., 0., 0., 0., 0.],  # sigma z
            [0.,  0.,  0., G, 0., 0. ],  # tau xy
            [0.,  0.,  0., 0., 0., 0.],  # tau zx
            [0.,  0.,  0., 0., 0., 0.],  # tau yz
        ], ti.f64)
        
        self.C = C; self.ti_C = ti_C
        self.C_6x6 = C_6x6; self.ti_C_6x6 = ti_C_6x6


class Linear_isotropic_planeStrain:
    def __init__(self, modulus: float, poisson_ratio: float):
        self.modulus = modulus; self.poisson_ratio = poisson_ratio
        self.dm = 2  # dimension for plane stress problem
        self.G = G = modulus / 2. / (1. + poisson_ratio)  # shear modulus
        term1 = modulus / (1. + poisson_ratio)
        term2 = poisson_ratio / (abs(1. - 2. * poisson_ratio) + 1.e-30)
        c00 = term1 * (1. + term2)
        c01 = term1 * term2
        C = np.array([    # elastic constants in Voigt notation, len(C) = dm + dm*(dm-1)/2, i.e. 2d -> 3, 3d -> 6
            [c00, c01, 0.],  # sigma x
            [c01, c00, 0.],  # sigma y
            [0.,  0.,  G ],  # tau xy
        ])
        ti_C = ti.Matrix([
            [c00, c01, 0.],  # sigma x
            [c01, c00, 0.],  # sigma y
            [0.,  0.,  G ],  # tau xy
        ], ti.f64)
        
        C_6x6 = np.array([  # voigt notation is related here, utilized to get 3D stress state for visulization
            [c00, c01, c01, 0., 0., 0.],  # sigma x
            [c01, c00, c01, 0., 0., 0.],  # sigma y
            [c01, c01, 0., 0., 0., 0.],  # sigma z
            [0.,  0.,  0., G, 0., 0. ],  # tau xy
            [0.,  0.,  0., 0., 0., 0.],  # tau zx
            [0.,  0.,  0., 0., 0., 0.],  # tau yz
        ])
        ti_C_6x6 = ti.Matrix([  # voigt notation is related here, utilized to get 3D stress state for visulization
            [c00, c01, c01, 0., 0., 0.],  # sigma x
            [c01, c00, c01, 0., 0., 0.],  # sigma y
            [c01, c01, 0., 0., 0., 0.],  # sigma z
            [0.,  0.,  0., G, 0., 0. ],  # tau xy
            [0.,  0.,  0., 0., 0., 0.],  # tau zx
            [0.,  0.,  0., 0., 0., 0.],  # tau yz
        ], ti.f64)

        self.C = C; self.ti_C = ti_C
        self.C_6x6 = C_6x6; self.ti_C_6x6 = ti_C_6x6