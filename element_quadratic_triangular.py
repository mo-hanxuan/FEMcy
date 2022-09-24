

import numpy as np
import taichi as ti

"""
some variables and functions related to quadratic triangular elements

        1
        |\
        | \
        4  3
        |   \
    η   |    \
    ^   2--5--0
    |
    ---> ξ 
"""
@ti.data_oriented
class Element_quadratic_triangular(object):
    def __init__(self, ):
        self.dm = 2  # spatial dimension for triangular element

        ## Gauss Points for quadratic triangle element
        self.gaussPoints = ti.Vector.field(self.dm, ti.f64, shape=(3, ))
        self.gaussPoints.from_numpy(np.array([
            [2./3., 1./6.],
            [1./6., 2./3.],
            [1./6., 1./6.],
        ]))
        self.gaussWeights = ti.field(dtype=ti.f64, shape=(3, ))
        self.gaussWeights.from_numpy(np.array([1./6., 1./6., 1./6.]))

        ### facets coordinates and normals for flux computation
        ### each facet has multiple gauss points
        ###     keys: tuple(id1, id2), use sorted tuple for convenience
        ###     values: natural coodinates of different Gauss points
        self.facet_natural_coos = {
            (0, 3): [[0.5, 0.5], [1., 0.]],
            (1, 3): [[0.5, 0.5], [0., 1.]],
            (1, 4): [[0., 0.5], [0., 1.]],
            (2, 4): [[0., 0.5], [0., 0.]],
            (2, 5): [[0.5, 0.], [0., 0.]],
            (0, 5): [[0.5, 0.], [1., 0.]],
        }
        self.facet_gauss_weights = {
            (0, 3): [0.5, 0.5], 
            (1, 3): [0.5, 0.5],
            (1, 4): [0.5, 0.5],
            (2, 4): [0.5, 0.5],
            (2, 5): [0.5, 0.5],
            (0, 5): [0.5, 0.5],
        }
        self.gaussPointNum_eachFacet = len(list(self.facet_gauss_weights.values())[0])

        """ facet normals in natural coordinate,
        must points to the outside of the element"""
        self.facet_natural_normals = {  
            ###     [[dξ, dη], ] each Gauss Point corresponds to a vector
            (0, 3): [[1., 1.], [1., 1.]],  
            (1, 3): [[1., 1.], [1., 1.]],
            (1, 4): [[-1., 0.], [-1., 0.]],
            (2, 4): [[-1., 0.], [-1., 0.]],
            (2, 5): [[0., -1.], [0., -1.]],
            (0, 5): [[0., -1.], [0., -1.]],
        }

        """facet number for reading .inp (Abaqus, CalculiX) file and get the face set"""
        self.inp_surface_num = [((0, 3), (3, 1)), 
                                ((1, 4), (4, 2)), 
                                ((2, 5), (5, 0))]

        """the nearest gauss point for each vertex, this is for mesh visualization, 
           where we can define color-per-vertex"""
        self.vertex_nearest_gaussPoint = ti.field(ti.i32, shape=(3))
        self.vertex_nearest_gaussPoint.from_numpy(np.array([1, 2, 0]))


    @ti.func
    def shapeFunc(self, natCoo):  
        """input the natural coordinate and get the shape function values"""
        nc = ti.Vector([natCoo[0], natCoo[1], 1. - natCoo[0] - natCoo[1]], ti.f64)
        return ti.Vector([
            nc[0] * (2. * nc[0] - 1.), 
            nc[1] * (2. * nc[1] - 1.), 
            nc[2] * (2. * nc[2] - 1.),
            4. * nc[0] * nc[1],
            4. * nc[1] * nc[2],
            4. * nc[2] * nc[0],
        ], ti.f64)


    @ti.func
    def dshape_dnat(self, natCoo):
        """derivative of shape function with respect to natural coodinate"""
        nc = ti.Vector([natCoo[0], natCoo[1], 1. - natCoo[0] - natCoo[1]], ti.f64)
        return ti.Matrix([
            [4.*nc[0] - 1., 0.],  # dshape0 / dnat
            [0., 4.*nc[1] - 1.],  # dshape1 / dnat
            [1. - 4.*nc[2], 1. - 4.*nc[2]],  # dshape2 / dnat
            [4.*nc[1], 4.*nc[0]],  # dshape3 / dnat
            [-4.*nc[1], 4.*(nc[2] - nc[1])],  # dshape4 / dnat
            [4.*(nc[2] - nc[0]), -4.*nc[0]]  # dshape5 / dnat
        ], ti.f64)


    def shapeFunc_pyscope(self, natCoo):  # triangle linear element
        """input the natural coordinate and get the shape function values"""
        nc = np.array([natCoo[0], natCoo[1], 1. - natCoo[0] - natCoo[1]])
        return np.array([
            nc[0] * (2. * nc[0] - 1.), 
            nc[1] * (2. * nc[1] - 1.), 
            nc[2] * (2. * nc[2] - 1.),
            4. * nc[0] * nc[1],
            4. * nc[1] * nc[2],
            4. * nc[2] * nc[0],
        ])


    def dshape_dnat_pyscope(self, natCoo):  # triangle linear element
        """derivative of shape function with respect to natural coodinate"""
        nc = np.array([natCoo[0], natCoo[1], 1. - natCoo[0] - natCoo[1]])
        return np.array([
            [4.*nc[0] - 1., 0.],  # dshape0 / dnat
            [0., 4.*nc[1] - 1.],  # dshape1 / dnat
            [1. - 4.*nc[2], 1. - 4.*nc[2]],  # dshape2 / dnat
            [4.*nc[1], 4.*nc[0]],  # dshape3 / dnat
            [-4.*nc[1], 4.*(nc[2] - nc[1])],  # dshape4 / dnat
            [4.*(nc[2] - nc[0]), -4.*nc[0]]  # dshape5 / dnat
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
        assert len(facet) == 2  # a facet defined by 2 nodes
        assert len(nodes) == 6  # 6 nodes of entire element
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
                e.g., m = 12 for qudritic triangular element
        """
        strain0 = np.array([dsdx[0, 0], 0.,
                            dsdx[1, 0], 0.,
                            dsdx[2, 0], 0.,
                            dsdx[3, 0], 0.,
                            dsdx[4, 0], 0.,
                            dsdx[5, 0], 0.])
        
        strain1 = np.array([0., dsdx[0, 1],
                            0., dsdx[1, 1],
                            0., dsdx[2, 1],
                            0., dsdx[3, 1],
                            0., dsdx[4, 1], 
                            0., dsdx[5, 1]])
        
        gammaxy = np.array([dsdx[0, 1], dsdx[0, 0],
                            dsdx[1, 1], dsdx[1, 0],
                            dsdx[2, 1], dsdx[2, 0],
                            dsdx[3, 1], dsdx[3, 0],
                            dsdx[4, 1], dsdx[4, 0],
                            dsdx[5, 1], dsdx[5, 0]])
        return np.array([strain0, strain1, gammaxy])
    

    @ti.func
    def strain_for_stiffnessMtrx_taichi(self, dsdx):
        return ti.Matrix([
            [dsdx[0, 0], 0.,
            dsdx[1, 0], 0.,
            dsdx[2, 0], 0.,
            dsdx[3, 0], 0.,
            dsdx[4, 0], 0.,
            dsdx[5, 0], 0.],

            [0., dsdx[0, 1],
            0., dsdx[1, 1],
            0., dsdx[2, 1],
            0., dsdx[3, 1],
            0., dsdx[4, 1], 
            0., dsdx[5, 1]],

            [dsdx[0, 1], dsdx[0, 0],
            dsdx[1, 1], dsdx[1, 0],
            dsdx[2, 1], dsdx[2, 0],
            dsdx[3, 1], dsdx[3, 0],
            dsdx[4, 1], dsdx[4, 0],
            dsdx[5, 1], dsdx[5, 0]]
        ], ti.float64)


    def show_triangles_2d(self, elements: np.ndarray, nodes: np.ndarray, 
                          surfaceEdges: np.ndarray, 
                          bottomleft: np.ndarray, stretchRatio: float,
                          refine=1):  # convert the quadratic triangle to 4 triangular parts
        """
        this function is called by show_2d in body.py
        input:
            elements: array of index of nodes of an element, shape = (-1, 6)
            nodes: nodes coordinates of original nodes

            refine: int, define the refinement depth for visualization
                refine=1 -> no refinement, just connect the in-edge nodes, 
                            cenvert the quadratic trangle to 4 linear triangles
                refine=2 -> each linear triangle is converted to 4 linear triangles
        output:
            a, b, c (np.ndarray) the three nodes of triangle
                for a & b & c, shape=(n, 2)
                     1st dimension is for different nodes
                     2nd dimension is for spatial dimension
            line0, line1: both begin nodes and end nodes of each line
        """
        ### the sequence should be the same with cases of 4 gauss points 
        ###     (each gauss point corresponds to a strain or stress value)
        tri_indexes = [  
            [4, 2, 5], [3, 5, 0], 
            [1, 4, 3], [4, 3, 5]
        ]
        e_num = elements.shape[0]
        # triangles = np.zeros((3, e_num * 4, 2))
        a, b, c = np.zeros((e_num * 4, 2)), np.zeros((e_num * 4, 2)), np.zeros((e_num * 4, 2))
        for iele, ele in enumerate(elements):
            for i_tri, tri in enumerate(tri_indexes):
                a[iele*4 + i_tri, :] = (nodes[ele[tri[0]], :] - bottomleft) * stretchRatio * 0.95
                b[iele*4 + i_tri, :] = (nodes[ele[tri[1]], :] - bottomleft) * stretchRatio * 0.95
                c[iele*4 + i_tri, :] = (nodes[ele[tri[2]], :] - bottomleft) * stretchRatio * 0.95
        
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
                (ele[0], ele[3], ele[5]), (ele[1], ele[3], ele[4]),
                (ele[2], ele[4], ele[5]), (ele[3], ele[4], ele[5]),
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
        
        noted: the Gauss points should be alligned at the order of the belowing figure
               so that the natural coordinates of outer nodes can be consistent with element constructed by the Gauss points
        1
        |\
        | \
        |  \ 
        |(1)\
        |    \    
        4     3
        |      \
        |       \
        |        \ 
        |(2)   (0)\
    η   |          \
    ^   2-----5-----0
    |
    ---> ξ 
        """
        natCoos = ti.Matrix([  # natural coordinates of outer nodes
            [5./3., -1./3., -1./3.],
            [-1./3., 5./3., -1./3.],
            [-1./3., -1./3., 5./3.],
            [2./3., 2./3., -1./3.],
            [-1./3., 2./3., 2./3.],
            [2./3., -1./3., 2./3.]
        ])
        for ele in nodal_vals:
            vec = ti.Vector([internal_vals[ele, i] for i in range(self.gaussPoints.shape[0])])
            nodal_vals[ele] = natCoos @ vec


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)

    ### test the global coordinates of the facet normals
    nodes = np.array([
        [0., 0.], 
        [30., 10.], 
        [35., 0.],
        [16, 4.],
        [32, 4],
        [16, 1]
    ])

    ELE = Element_quadratic_triangular()

    print("element_triangle_linear.global_normal(nodes, [0, 3]) = ", 
          ELE.global_normal(nodes, [0, 3]))
    print("element_triangle_linear.global_normal(nodes, [3, 1]) = ", 
          ELE.global_normal(nodes, [3, 1]))
    print("element_triangle_linear.global_normal(nodes, [1, 4]) = ", 
          ELE.global_normal(nodes, [1, 4]))
    print("element_triangle_linear.global_normal(nodes, [4, 2]) = ", 
          ELE.global_normal(nodes, [4, 2]))