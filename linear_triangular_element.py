
import numpy as np
import taichi as ti

"""
some variables and functions related to linear triangular elements

        1
        |\
        |  \
    η   |    \
    ^   2 --- 0
    |
    ---> ξ 
"""
@ti.data_oriented
class Linear_triangular_element(object):
    def __init__(self, ):
        self.dm = 2  # spatial dimension for triangular element

        ### Gauss points' coordinates (natural) and weights, for linear triangle element
        self.gaussPoints = ti.Vector.field(self.dm, ti.f64, shape=(1, ))  # reduce integration, valid for lineat triangle element
        self.gaussPoints.from_numpy(np.array([[1./3., 1./3.], ]))
        self.gaussWeights = ti.field(dtype=ti.f64, shape=(1, ))
        self.gaussWeights.from_numpy(np.array([1./2., ]))
        self.gaussPointNum_eachFacet = 1
        self.gaussPoints_visualize = self.gaussPoints

        ### facets coordinates and normals for flux computation
        ### each facet has multiple gauss points
        ###     keys: tuple(id1, id2), use sorted tuple for convenience
        ###     values: natural coodinates of different Gauss points
        self.facet_natural_coos = {
            (0, 1): [[0.5, 0.5], ], 
            (1, 2): [[0., 0.5], ],  # only one gauss points for each facet
            (0, 2): [[0.5, 0.], ],
        }
        self.facet_gauss_weights = {
            (0, 1): [1., ], 
            (1, 2): [1., ],  # only one gauss points for each facet
            (0, 2): [1., ],
        }

        """ facet normals in natural coordinate,
        must points to the outside of the element"""
        self.facet_natural_normals = {  
            ###     [[dξ, dη], ] each Gauss Point corresponds to a vector
            (0, 1): [[2**0.5/2., 2**0.5/2.], ], 
            (1, 2): [[-1., 0.], ],  
            (0, 2): [[0., -1.], ],
        }

        """facet number for reading .inp (Abaqus, CalculiX) file and get the face set"""
        self.inp_surface_num = [((0, 1), ),  # the counterpart in quadratic element is ((0, 3), (3, 1))
                                ((1, 2), ), 
                                ((2, 0), )]

        """the nearest gauss point for each vertex, this is for mesh visualization, 
           where we can define color-per-vertex"""
        self.vertex_nearest_gaussPoint = ti.field(ti.i32, shape=(3))
        self.vertex_nearest_gaussPoint.from_numpy(np.array([0, 0, 0]))


    @ti.func
    def shapeFunc(self, natCoo):  # triangle linear element
        """input the natural coordinate and get the shape function values"""
        return ti.Vector([natCoo[0], natCoo[1], 1. - natCoo[0] - natCoo[1]], ti.f64)

    @ti.func
    def dshape_dnat(self, natCoo):  # triangle linear element
        """derivative of shape function with respect to natural coodinate"""
        return ti.Matrix([
            [1., 0.],  # dshape0 / dnat
            [0., 1.],  # dshape1 / dnat
            [-1., -1.]  # dshape2 / dnat
        ], ti.f64)


    def shapeFunc_pyscope(self, natCoo):  # triangle linear element
        """input the natural coordinate and get the shape function values"""
        return np.array([natCoo[0], natCoo[1], 1. - natCoo[0] - natCoo[1]])

    def dshape_dnat_pyscope(self, natCoo):  # triangle linear element
        """derivative of shape function with respect to natural coodinate"""
        return np.array([
            [1., 0.],  # dshape0 / dnat
            [0., 1.],  # dshape1 / dnat
            [-1., -1.]  # dshape2 / dnat
        ])


    def global_normal(self, nodes: np.ndarray, facet: list, gaussPointId=0):
        """
        deduce the normal vector in global coordinate for a given facet.
        input:
            nodes: global coordinates of all nodes of this element,
            facet: local node-idexes of the given facet
            gaussPointId: the index of the gauss point of this facet
                            the archetecture here can be generalized to multiple Gauss points
        output: 
            global coordinates of this little facet, 
            where the length of the vector indicates the area of the boundary facet
        """
        assert len(facet) == 2  # 2 nodes on a facet of triangle linear element
        assert len(nodes) == 3  # 3 nodes of entire element
        facet = tuple(sorted(facet))
        
        ### facet normal from natural coordinates to global coordinates,  
        ### must maintain the perpendicular relation between normal and facet, 
        ### thus, the operation is: n_g = n_t @ (dx/dξ)^(-1)
        natCoo = self.facet_natural_coos[facet][gaussPointId]
        dsdn = self.dshape_dnat_pyscope(natCoo)
        dxdn = nodes.transpose() @ dsdn

        natural_normal = self.facet_natural_normals[facet][gaussPointId]
        global_normal = natural_normal @ np.linalg.inv(dxdn)  # n_g = n_t @ (dx/dξ)^(-1)
        global_normal /= np.linalg.norm(global_normal) + 1.e-30  # normalize

        ### multiply the boundary size and Gauss Weight
        area_x_gaussWeight = np.linalg.norm(nodes[facet[0]] - nodes[facet[1]]) * \
                                self.facet_gauss_weights[facet][gaussPointId]

        return global_normal, area_x_gaussWeight


    def strain_for_stiffnessMtrx(self, dsdx: np.ndarray):
        """
        strain for the stiffness matrix:
        with shape = (n, m), 
            n is the dimension of strian in Voigt notation, 
                e.g., n = 3 for components including epsilon_11, epsilon_22 and gamma_12 (= 2 * epsilon_12)
            m is the number of dof of this element, 
                e.g., m = 6 for (node0_u0, node0_u1, node1_u0, node1_u1, node2_u0, node2_u1)
        """
        strain0 = np.array([dsdx[0, 0], 0.,
                            dsdx[1, 0], 0.,
                            dsdx[2, 0], 0.])
        strain1 = np.array([0., dsdx[0, 1],
                            0., dsdx[1, 1],
                            0., dsdx[2, 1]])
        gammaxy = np.array([dsdx[0, 1], dsdx[0, 0],
                            dsdx[1, 1], dsdx[1, 0],
                            dsdx[2, 1], dsdx[2, 0]])
        return np.array([strain0, strain1, gammaxy])


    @ti.func
    def strain_for_stiffnessMtrx_taichi(self, dsdx):
        """
        strain for the stiffness matrix:
        with shape = (n, m), 
            n is the dimension of strian in Voigt notation, 
                e.g., n = 3 for components including epsilon_11, epsilon_22 and gamma_12 (= 2 * epsilon_12)
            m is the number of dof of this element, 
                e.g., m = 6 for (node0_u0, node0_u1, node1_u0, node1_u1, node2_u0, node2_u1)
        """
        return ti.Matrix([
            [dsdx[0, 0], 0.,
            dsdx[1, 0], 0.,
            dsdx[2, 0], 0.],  # strain0
            
            [0., dsdx[0, 1],
            0., dsdx[1, 1],
            0., dsdx[2, 1]], # strain1
            
            [dsdx[0, 1], dsdx[0, 0],
            dsdx[1, 1], dsdx[1, 0],
            dsdx[2, 1], dsdx[2, 0]],  # gammaxy
        ], ti.f64)


    def show_triangles_2d(self, elements: np.ndarray, nodes: np.ndarray, 
                          surfaceEdges: np.ndarray, 
                          bottomleft: np.ndarray, stretchRatio: float,
                        ):  # convert the quadratic triangle to 4 triangular parts
        """
        this function is called by show_2d in body.py
        input:
            elements: array of index of nodes of an element, shape = (-1, 3)
            nodes: nodes coordinates of original nodes
        output:
            a, b, c (np.ndarray) the three nodes of triangle
                for a & b & c, shape=(n, 2)
                     1st dimension is for different nodes
                     2nd dimension is for spatial dimension
            line0, line1: both begin nodes and end nodes of each line
        """
        vertices = nodes
        triangles = elements
        vertices = (vertices - bottomleft) * stretchRatio * 0.95  # in GUI, you need to normalize the coordinates to 0~1
        a = np.array([vertices[nodes[0]] for nodes in triangles])
        b = np.array([vertices[nodes[1]] for nodes in triangles])
        c = np.array([vertices[nodes[2]] for nodes in triangles])

        line0 = np.array([nodes[line[0]] for line in surfaceEdges])
        line1 = np.array([nodes[line[1]] for line in surfaceEdges])
        line0 = (line0 - bottomleft) * stretchRatio * 0.95
        line1 = (line1 - bottomleft) * stretchRatio * 0.95
        return a, b, c, line0, line1


    def get_mesh(self, elements: np.ndarray):
        """get the triangles of the mesh, and the outer surface of the mesh
           return: 
                mesh: a int array of shape (3 * #triangles), which indicate
                    the vertex indices of the triangles. 
                face2ele: a dict with keys for face (sorted tuple), 
                    and values of set of elements sharing this face
                surfaces: the triangle surface that you want to visualize, 
                    if for 2d case, you can just let all triangles to be the surface
        """
        mesh = set()
        face2ele = {}  # from face to element
        for iele, ele in enumerate(elements):
            faces = [ 
                (ele[0], ele[1], ele[2]), 
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


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)

    ### test the global coordinates of the facet normals
    nodes = np.array([
        [0., 0.], 
        [30., 10.], 
        [35., 0.],
    ])

    ELE = Linear_triangular_element()

    print("element_triangle_linear.global_normal(nodes, [1, 0]) = ", 
          ELE.global_normal(nodes, [1, 0]))
    print("element_triangle_linear.global_normal(nodes, [1, 2]) = ", 
          ELE.global_normal(nodes, [1, 2]))
    print("element_triangle_linear.global_normal(nodes, [0, 2]) = ", 
          ELE.global_normal(nodes, [0, 2]))