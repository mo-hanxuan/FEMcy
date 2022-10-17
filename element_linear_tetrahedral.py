import numpy as np
import taichi as ti

"""  the shape and node number of a tetrahedral element (C3D4 element)
                3
               /| \
              / |   \
             /  |     \ 
            /   2------1
           /   *     /**
   η      /  *  / **
   ^     0 / **
   |
   ---> ξ
  /
 ζ
""" 
@ti.data_oriented
class Element_linear_tetrahedral(object):
    def __init__(self, gauss_points_count=1):
        self.dm = 3  # spatial dimensions

        ### Gauss points for linear tetrahedral element
        self.gaussPoints = ti.Vector.field(self.dm, ti.f64, shape=(1,))
        self.gaussPoints.from_numpy(np.array([[0.25, 0.25, 0.25], ]))
        self.gaussWeights = ti.field(dtype=ti.f64, shape=(1, ))
        self.gaussWeights.from_numpy(np.array([1./6., ]))
        self.integPointNum_eachFacet = 1
        self.gaussPoints_visualize = self.gaussPoints

        ### facets coordinates and normals for flux computation
        ### each facet has multiple gauss points
        ###     keys: tuple(id1, id2), use sorted tuple for convenience
        ###     values: natural coodinates of different Gauss points
        self.facet_natural_coos = {
            (1, 2, 3): [[1./3., 1./3., 0.], ],
            (0, 2, 3): [[0., 1./3., 1./3.], ],
            (0, 1, 3): [[1./3., 1./3., 1./3.], ],
            (0, 1, 2): [[1./3., 0., 1./3.], ], 
        }
        self.facet_point_weights = {
            (1, 2, 3): [1., ],
            (0, 2, 3): [1., ],
            (0, 1, 3): [1., ],
            (0, 1, 2): [1., ], 
        }

        """ facet normals in natural coordinate,
        must points to the outside of the element"""
        self.facet_natural_normals = {  
            ###        [[dξ, dη, dζ], ],
            (1, 2, 3): [[0., 0., -1.], ],
            (0, 2, 3): [[-1., 0., 0.], ],
            (0, 1, 3): [[1., 1., 1.], ],
            (0, 1, 2): [[0., -1., 0.], ], 
        }
        """facet number for reading .inp (Abaqus, CalculiX) file and get the face set"""
        self.inp_surface_num = [((0, 1, 2), ), 
                                ((0, 1, 3), ), 
                                ((1, 2, 3), ),
                                ((0, 2, 3), )]


    @ti.func
    def shapeFunc(self, natCoo):
        """input the natural coordinate and get the shape function values"""
        return ti.Vector([natCoo[2], natCoo[0], 1. - natCoo[0] - natCoo[1] - natCoo[2], 
                          natCoo[1]], ti.f64)


    @ti.func
    def dshape_dnat(self, natCoo):  
        """derivative of shape function with respect to natural coodinate"""
        return ti.Matrix([
            [0., 0., 1.],  # dshape0 / dnat
            [1., 0., 0.],  # dshape1 / dnat
            [-1., -1., -1.], # dshape2 / dnat
            [0., 1., 0.]  # dshape3 / dnat
        ], ti.f64)
    

    def shapeFunc_pyscope(self, natCoo):
        """input the natural coordinate and get the shape function values"""
        return np.array([natCoo[2], natCoo[0], 1. - natCoo[0] - natCoo[1] - natCoo[2], 
                          natCoo[1]])


    def dshape_dnat_pyscope(self, natCoo):  
        """derivative of shape function with respect to natural coodinate"""
        return np.array([
            [0., 0., 1.],  # dshape0 / dnat
            [1., 0., 0.],  # dshape1 / dnat
            [-1., -1., -1.], # dshape2 / dnat
            [0., 1., 0.]  # dshape3 / dnat
        ])


    def global_normal(self, nodes: np.ndarray, facet: list, integPointId=0):
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
            assert len(facet) == 3  # 3 nodes on a facet of triangle tetrahedral element
            assert len(nodes) == 4  # 4 nodes of entire element
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
                e.g., m = 12 (= 4 x 3) for 4 nodes with dm = 3 for each nodes
        """
        return ti.Matrix([
            [dsdx[0, 0], 0., 0., 
             dsdx[1, 0], 0., 0., 
             dsdx[2, 0], 0., 0.,
             dsdx[3, 0], 0., 0.,],  # strain0
            
            [0., dsdx[0, 1], 0.,
             0., dsdx[1, 1], 0.,
             0., dsdx[2, 1], 0.,
             0., dsdx[3, 1], 0.,], # strain1

            [0.,  0., dsdx[0, 2],
             0.,  0., dsdx[1, 2], 
             0.,  0., dsdx[2, 2], 
             0.,  0., dsdx[3, 2], ], # strain2
            
            [dsdx[0, 1], dsdx[0, 0], 0.,
             dsdx[1, 1], dsdx[1, 0], 0., 
             dsdx[2, 1], dsdx[2, 0], 0.,
             dsdx[3, 1], dsdx[3, 0], 0.,],  # gamma_01, 

            [dsdx[0, 2], 0., dsdx[0, 0], 
             dsdx[1, 2], 0., dsdx[1, 0], 
             dsdx[2, 2], 0., dsdx[2, 0], 
             dsdx[3, 2], 0., dsdx[3, 0], ],  # gamma_20, 

            [0., dsdx[0, 2], dsdx[0, 1], 
             0., dsdx[1, 2], dsdx[1, 1], 
             0., dsdx[2, 2], dsdx[2, 1], 
             0., dsdx[3, 2], dsdx[3, 1], ],  # gamma_12, 
        ], ti.f64)
    

    def get_mesh(self, elements: np.ndarray):
        """get the triangles of the mesh, and the outer surface of the mesh"""
        mesh = set()
        face2ele = {}  # from face to element
        for iele, ele in enumerate(elements):
            faces = [ 
                tuple(sorted((ele[1], ele[2], ele[3]))), 
                tuple(sorted((ele[0], ele[2], ele[3]))),
                tuple(sorted((ele[0], ele[1], ele[3]))), 
                tuple(sorted((ele[0], ele[1], ele[2])))
            ]
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
        
        noted: for linear element, we simply just extrapolate the center point to all nodes
        """
        for ele in nodal_vals:
            nodal_vals[ele].fill(internal_vals[ele, 0])