# yapf: disable

import numpy as np
import taichi as ti
from element_base import ElementBase

"""  the shape and node number of a tetrahedral element (C3D4 element)
                    3
                   /| \
                  / |   \
                 /  |     \ 
                /   9      8
               7    |        \
              /     |          \
             /      |            \
            /       2------5------1 
           /    ,/          ,,,/**
          /   6         4
   η     / ,/   ,,/***
   ^    0
   |
   ---> ξ
  /
 ζ
"""
@ti.data_oriented
class Element_quadratic_tetrahedral(ElementBase):
    def __init__(self, gauss_points_count=4):
        self.dm = 3  # spatial dimension

        """ get the gauss point, 
            point's position refers to https://help.febio.org/FEBio/FEBio_tm_2_7/FEBio_tm_2-7-Subsection-4.1.4.html"""
        a = 0.585410196624968; b = 0.138196601125010
        self.gaussPoints = ti.Vector.field(self.dm, ti.f64, shape=(4, ))
        self.gaussPoints.from_numpy(np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ]))
        self.gaussWeights = ti.field(dtype=ti.f64, shape=(4, ))
        self.gaussWeights.from_numpy(np.array([1./24., 1./24., 1./24., 1./24.]))

        """ facets coordinates and normals for flux computation
            each facet can have multiple gauss points in general
                keys: tuple(id1, id2), use sorted tuple for convenience
                values: natural coodinates of different Gauss points"""
        self.facet_natural_coos = {
            (1, 2, 3, 5, 8, 9): [[1., 0., 0.], [0., 1., 0.], [0., 0., 0.],
                                 [0.5, 0.5, 0.], [0., 0.5, 0.], [0.5, 0., 0.]],  # face (1, 2, 3)
            (0, 2, 3, 6, 7, 9): [[0., 1., 0.], [0., 0., 1.], [0., 0., 0.],
                                 [0., 0.5, 0.], [0., 0.5, 0.5], [0., 0., 0.5]],  # face (0, 2, 3)
            (0, 1, 3, 4, 7, 8): [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                 [0.5, 0., 0.5], [0., 0.5, 0.5], [0.5, 0.5, 0.]],  # face (0, 1, 3)
            (0, 1, 2, 4, 5, 6): [[1., 0., 0.], [0., 0., 1.], [0., 0., 0.],
                                 [0.5, 0., 0.5], [0.5, 0., 0.], [0., 0., 0.5]],  # face (0, 1, 2)
        }
        self.facet_point_weights = {
            (1, 2, 3, 5, 8, 9): [1./12., 1./12., 1./12., 1./4., 1./4., 1./4.],
            (0, 2, 3, 6, 7, 9): [1./12., 1./12., 1./12., 1./4., 1./4., 1./4.],
            (0, 1, 3, 4, 7, 8): [1./12., 1./12., 1./12., 1./4., 1./4., 1./4.],
            (0, 1, 2, 4, 5, 6): [1./12., 1./12., 1./12., 1./4., 1./4., 1./4.],
        }
        self.integPointNum_eachFacet = len(list(self.facet_point_weights.values())[0])

        """ facet normals in natural coordinate,
            must points to the outside of the element"""
        self.facet_natural_normals = {
            ###                 [[dξ,  dη, dζ], ],
            (1, 2, 3, 5, 8, 9): [[0., 0., -1.], [0., 0., -1.], [0., 0., -1.],
                                 [0., 0., -1.], [0., 0., -1.], [0., 0., -1.]],  # (1, 2, 3)
            (0, 2, 3, 6, 7, 9): [[-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.],
                                 [-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]],  # (0, 2, 3)
            (0, 1, 3, 4, 7, 8): [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.],
                                 [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],  # (0, 1, 3)
            (0, 1, 2, 4, 5, 6): [[0., -1., 0.], [0., -1., 0.], [0., -1., 0.],
                                 [0., -1., 0.], [0., -1., 0.], [0., -1., 0.]],  # (0, 1, 2)
        }
        """facet number for reading .inp (Abaqus, CalculiX) file and get the face set"""
        self.inp_surface_num = [((0, 1, 2, 4, 5, 6), ),  # (0, 1, 2) 
                                ((0, 1, 3, 4, 7, 8), ),  # (0, 1, 3) 
                                ((1, 2, 3, 5, 8, 9), ),  # (1, 2, 3)
                                ((0, 2, 3, 6, 7, 9), )]  # (0, 2, 3)


    @ti.func
    def shapeFunc(self, natCoo):
        """input the natural coordinate and get the shape function values"""
        nc = ti.Vector([natCoo[2],  # nc[0]
                        natCoo[0],  # nc[1]
                        1. - natCoo[0] - natCoo[1] - natCoo[2],  # nc[2]
                        natCoo[1],  # nc[3]
                       ], ti.f64)
        return ti.Vector([
            nc[0] * (2. * nc[0] - 1.),
            nc[1] * (2. * nc[1] - 1.),
            nc[2] * (2. * nc[2] - 1.),
            nc[3] * (2. * nc[3] - 1.),
            4. * nc[0] * nc[1],
            4. * nc[1] * nc[2],
            4. * nc[2] * nc[0],
            4. * nc[0] * nc[3],
            4. * nc[3] * nc[1],
            4. * nc[2] * nc[3],
        ], ti.f64)


    @ti.func
    def dshape_dnat(self, natCoo):
        """derivative of shape function with respect to natural coodinate"""
        nc = ti.Vector([natCoo[2], natCoo[0],
                        1. - natCoo[0] - natCoo[1] - natCoo[2],
                        natCoo[1]], ti.f64)
        return ti.Matrix([
            [0., 0., 4.*nc[0] - 1.],  # dshape0 / dnat
            [4.*nc[1] - 1., 0., 0.],  # dshape1 / dnat
            [1. - 4.*nc[2], 1. - 4.*nc[2], 1. - 4.*nc[2]],  # dshape2 / dnat
            [0., 4.*nc[3] - 1., 0.],  # dshape3 / dnat

            [4.*nc[0], 0., 4.*nc[1]],  # dshape4 / dnat
            [4.*(nc[2] - nc[1]), -4.*nc[1], -4.*nc[1]],  # dshape5 / dnat
            [-4.*nc[0], -4.*nc[0], 4.*(nc[2] - nc[0])],  # dshape6 / dnat
            [0., 4.*nc[0], 4.*nc[3]],   # dshape7 / dnat
            [4.*nc[3], 4.*nc[1], 0.],   # dshape8 / dnat
            [-4.*nc[3], 4.*(nc[2] - nc[3]), -4.*nc[3]],  # dshape9 / dnat
        ], ti.f64)


    def shapeFunc_pyscope(self, natCoo):  # triangle linear element
        """input the natural coordinate and get the shape function values"""
        nc = np.array([natCoo[2],  # nc[0]
                        natCoo[0],  # nc[1]
                        1. - natCoo[0] - natCoo[1] - natCoo[2],  # nc[2]
                        natCoo[1],  # nc[3]
                       ])
        return np.array([
            nc[0] * (2. * nc[0] - 1.),
            nc[1] * (2. * nc[1] - 1.),
            nc[2] * (2. * nc[2] - 1.),
            nc[3] * (2. * nc[3] - 1.),
            4. * nc[0] * nc[1],
            4. * nc[1] * nc[2],
            4. * nc[2] * nc[0],
            4. * nc[0] * nc[3],
            4. * nc[3] * nc[1],
            4. * nc[2] * nc[3],
        ])


    def dshape_dnat_pyscope(self, natCoo):  # triangle linear element
        """derivative of shape function with respect to natural coodinate"""
        nc = np.array([natCoo[2], natCoo[0],
                        1. - natCoo[0] - natCoo[1] - natCoo[2],
                        natCoo[1]])
        return np.array([
            [0., 0., 4.*nc[0] - 1.],  # dshape0 / dnat
            [4.*nc[1] - 1., 0., 0.],  # dshape1 / dnat
            [1. - 4.*nc[2], 1. - 4.*nc[2], 1. - 4.*nc[2]],  # dshape2 / dnat
            [0., 4.*nc[3] - 1., 0.],  # dshape3 / dnat

            [4.*nc[0], 0., 4.*nc[1]],  # dshape4 / dnat
            [4.*(nc[2] - nc[1]), -4.*nc[1], -4.*nc[1]],  # dshape5 / dnat
            [-4.*nc[0], -4.*nc[0], 4.*(nc[2] - nc[0])],  # dshape6 / dnat
            [0., 4.*nc[0], 4.*nc[3]],   # dshape7 / dnat
            [4.*nc[3], 4.*nc[1], 0.],   # dshape8 / dnat
            [-4.*nc[3], 4.*(nc[2] - nc[3]), -4.*nc[3]],  # dshape9 / dnat
        ])


    def globalNormal(self, nodes: np.ndarray, facet: list, integPointId=0):
        """
            deduce the normal vector in global coordinate for a given facet.
            input:
                nodes: global coordinates of all nodes of this element,
                facet: local node-idexes of the given facet
                integPointId: the index of the gauss point of this facet
                                the archetecture here can be generalized to multiple Gauss points
            output: 
                global coordinates of this little facet, 
                where the length of the vector indicates the area of the boundary facet
            """
        facet = tuple(sorted(facet))

        ### facet normal from natural coordinates to global coordinates,
        ### must maintain the perpendicular relation between normal and facet,
        ### thus, the operation is: n_g = n_t @ (dx/dξ)^(-1)
        natCoo = self.facet_natural_coos[facet][integPointId]
        dsdn = self.dshape_dnat_pyscope(natCoo)
        dxdn = nodes.transpose() @ dsdn

        natural_normal = self.facet_natural_normals[facet][integPointId]
        global_normal = natural_normal @ np.linalg.inv(dxdn)  # n_g = n_t @ (dx/dξ)^(-1)
        global_normal /= np.linalg.norm(global_normal) + 1.e-30  # normalize

        ### multiply the boundary size and Gauss Weight
        area = np.cross(nodes[facet[1]] - nodes[facet[0]],
                        nodes[facet[2]] - nodes[facet[0]])
        area = 0.5 * np.linalg.norm(area)
        area_x_weight = area * self.facet_point_weights[facet][integPointId]

        return global_normal, area_x_weight


    @ti.func
    def strainMtrx(self, dsdx):
        """
        strain for the stiffness matrix:
        with shape = (n, m), 
            n is the dimension of strian in Voigt notation, 
                e.g., n = 6 for components including ε00, ε11, ε22, γ01, γ20, γ12 (= 2 * ε12)
            m is the number of dof of this element, 
                e.g., m = 30 (10 nodes x 3-dimension) for qudratic tetrahedral element
        """
        return ti.Matrix([
            [dsdx[0, 0], 0., 0.,    dsdx[1, 0], 0., 0.,
             dsdx[2, 0], 0., 0.,    dsdx[3, 0], 0., 0.,
             dsdx[4, 0], 0., 0.,    dsdx[5, 0], 0., 0.,
             dsdx[6, 0], 0., 0.,    dsdx[7, 0], 0., 0.,
             dsdx[8, 0], 0., 0.,    dsdx[9, 0], 0., 0., ],  # strain0

            [0., dsdx[0, 1], 0.,    0., dsdx[1, 1], 0.,
             0., dsdx[2, 1], 0.,    0., dsdx[3, 1], 0.,
             0., dsdx[4, 1], 0.,    0., dsdx[5, 1], 0.,
             0., dsdx[6, 1], 0.,    0., dsdx[7, 1], 0.,
             0., dsdx[8, 1], 0.,    0., dsdx[9, 1], 0.,], # strain1

            [0., 0., dsdx[0, 2],    0., 0., dsdx[1, 2],
             0., 0., dsdx[2, 2],    0., 0., dsdx[3, 2],
             0., 0., dsdx[4, 2],    0., 0., dsdx[5, 2],
             0., 0., dsdx[6, 2],    0., 0., dsdx[7, 2],
             0., 0., dsdx[8, 2],    0., 0., dsdx[9, 2],], # strain2

            [dsdx[0, 1], dsdx[0, 0], 0.,    dsdx[1, 1], dsdx[1, 0], 0.,
             dsdx[2, 1], dsdx[2, 0], 0.,    dsdx[3, 1], dsdx[3, 0], 0.,
             dsdx[4, 1], dsdx[4, 0], 0.,    dsdx[5, 1], dsdx[5, 0], 0.,
             dsdx[6, 1], dsdx[6, 0], 0.,    dsdx[7, 1], dsdx[7, 0], 0.,
             dsdx[8, 1], dsdx[8, 0], 0.,    dsdx[9, 1], dsdx[9, 0], 0.,],  # gamma_01, 

            [dsdx[0, 2], 0., dsdx[0, 0],    dsdx[1, 2], 0., dsdx[1, 0],
             dsdx[2, 2], 0., dsdx[2, 0],    dsdx[3, 2], 0., dsdx[3, 0],
             dsdx[4, 2], 0., dsdx[4, 0],    dsdx[5, 2], 0., dsdx[5, 0],
             dsdx[6, 2], 0., dsdx[6, 0],    dsdx[7, 2], 0., dsdx[7, 0],
             dsdx[8, 2], 0., dsdx[8, 0],    dsdx[9, 2], 0., dsdx[9, 0],],  # gamma_20, 

            [0., dsdx[0, 2], dsdx[0, 1],    0., dsdx[1, 2], dsdx[1, 1],
             0., dsdx[2, 2], dsdx[2, 1],    0., dsdx[3, 2], dsdx[3, 1],
             0., dsdx[4, 2], dsdx[4, 1],    0., dsdx[5, 2], dsdx[5, 1],
             0., dsdx[6, 2], dsdx[6, 1],    0., dsdx[7, 2], dsdx[7, 1],
             0., dsdx[8, 2], dsdx[8, 1],    0., dsdx[9, 2], dsdx[9, 1],],  # gamma_12, 
        ], ti.f64)


    def getMesh(self, elements: np.ndarray):
        """get the triangles of the mesh, and the outer surface of the mesh"""
        mesh = set()
        face2ele = {}  # from face to element
        for iele, ele in enumerate(elements):
            faces = [
                # ele[1], ele[2], ele[3],
                (ele[1], ele[5], ele[8]), (ele[3], ele[8], ele[9]),
                (ele[2], ele[5], ele[9]), (ele[5], ele[9], ele[8]),

                # ele[0], ele[2], ele[3],
                (ele[0], ele[6], ele[7]), (ele[3], ele[7], ele[9]),
                (ele[2], ele[9], ele[6]), (ele[6], ele[7], ele[9]),

                # ele[0], ele[1], ele[3],
                (ele[0], ele[4], ele[7]), (ele[1], ele[8], ele[4]),
                (ele[3], ele[7], ele[8]), (ele[4], ele[7], ele[8]),

                # ele[0], ele[1], ele[2]
                (ele[0], ele[4], ele[6]), (ele[1], ele[5], ele[4]),
                (ele[2], ele[6], ele[5]), (ele[4], ele[5], ele[6]),
            ]
            faces = list(map(lambda face: tuple(sorted(face)), faces))
            for face in faces:
                mesh.add(face)
                if face in face2ele:
                    face2ele[face].add(iele)
                else:
                    face2ele[face] = {iele}
        ### get the surface mesh
        surfaces = set()
        for face in face2ele:
            if len(face2ele[face]) == 1:
                surfaces.add(face)

        mesh = np.array(list(mesh)); surfaces = np.array(list(surfaces))
        return mesh, face2ele, surfaces


    @ti.kernel
    def extrapolate(self, internal_vals: ti.template(), nodal_vals: ti.template()):
        """extrapolate the internal Gauss points' vals to the nodal vals
           no averaging is performing here, we want to get the nodal vals patch by patch,
           so the each patch maintain the original values at Gauss points, but different patches
           have different values at their share nodes
        input:
            internal_vals: scaler field with shape = (elements.shape[0], gaussPoints.shape[0])
            nodal_vals: vector field with shape = (elements.shape[0],), and the vector has dimension of elements.shape[0]
        update:
            nodal_vals is updated after this function
        
        noted: gauss points' number is written in bracket 
               the Gauss points should be alligned at the order of the belowing figure
               so that the natural coordinates of outer nodes can be consistent with element constructed by the Gauss points
                                3
                               /| \
                              / |   \
                             /  |(1)  \ 
                            /   9      8
                           7    |        \
                          /     |          \
                         /      |(3)    (0)  \
                        /       2------5------1 
                       / (2),/          ,,,/**
                      /   6         4
               η     / ,/   ,,/***
               ^    0
               |
               ---> ξ
              /
             ζ
        """
        a = 0.585410196624968; b = 0.138196601125010
        c = (1. - a) / (a - b); d = b / (a - b)
        x = (0.5 - b) / (a - b)
        natCoos = ti.Matrix([  # natural coordinates of outer nodes
            [-d, -d, 1. + c, 2.*d-c],  # 0
            [1. + c, -d, -d, 2.*d-c],  # 1
            [-d, -d, -d, 1.+3.*d],  # 2
            [-d, 1. + c, -d, 2.*d-c],  # 3
            [x, -d, x, 1.-2.*x+d],  # 4
            [x, -d, -d, 1.+2.*d-x],  # 5
            [-d, -d, x, 1.+2.*d-x],  # 6
            [-d, x, x, 1.-2.*x+d],  # 7
            [x, x, -d, 1.-2.*x+d],  # 8
            [-d, x, -d, 1.+2.*d-x],  # 9
        ])
        for ele in nodal_vals:
            vec = ti.Vector([internal_vals[ele, i] for i in range(self.gaussPoints.shape[0])])
            nodal_vals[ele] = natCoos @ vec
