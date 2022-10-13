import numpy as np
import taichi as ti

"""
        3------2
        |      |
    η   |      | 
    ^   0------1
    |
    ---> ξ 
"""
@ti.data_oriented
class Element_linear_quadrilateral(object):
    def __init__(self, ):
        self.dm = 2  # spatial dimension

        self.gaussPoints = ti.Vector.field(self.dm, ti.f64, shape=(4, ))
        tmp = 1. / 3.**0.5
        self.gaussPoints.from_numpy(np.array([
            [-tmp, -tmp],
            [tmp, -tmp],
            [tmp, tmp],
            [-tmp, tmp],
        ]))
        self.gaussWeights = ti.field(dtype=ti.f64, shape=(4, ))
        self.gaussWeights.from_numpy(np.array([1., 1., 1., 1.]))

        ### facets coordinates and normals for flux computation
        ### each facet has multiple gauss points
        ###     keys: tuple(id1, id2), use sorted tuple for convenience
        ###     values: natural coodinates of different Gauss points
        self.facet_natural_coos = {
            (0, 1): [[-1., -1.], [1., -1.]],
            (1, 2): [[1., -1.], [1., 1.]],
            (2, 3): [[1., 1.], [-1., 1.]],
            (0, 3): [[-1., 1.], [-1., -1.]] 
        }
        self.facet_point_weights = {
            (0, 1): [0.5, 0.5],
            (1, 2): [0.5, 0.5],
            (2, 3): [0.5, 0.5],
            (0, 3): [0.5, 0.5] 
        }
        self.integPointNum_eachFacet = len(list(self.facet_point_weights.values())[0])

        """ facet normals in natural coordinate,
        must points to the outside of the element"""
        self.facet_natural_normals = {  
            ###     [[dξ, dη], ] each Gauss Point corresponds to a vector
            (0, 1): [[0., -1.], [0., -1.]],
            (1, 2): [[1., 0.], [1., 0.]],
            (2, 3): [[0., 1.], [0., 1.]],
            (0, 3): [[-1., 0.], [-1., 0.]] 
        }

        """facet number for reading .inp (Abaqus, CalculiX) file and get the face set"""
        self.inp_surface_num = [((0, 1), ), 
                                ((1, 2), ), 
                                ((2, 3), ),
                                ((0, 3), )]
        

    @ti.func
    def shapeFunc(self, natCoo):  
        """input the natural coordinate and get the shape function values"""
        return ti.Vector([
            (1. - natCoo[0]) * (1. - natCoo[1]) / 4.,
            (1. + natCoo[0]) * (1. - natCoo[1]) / 4.,
            (1. + natCoo[0]) * (1. + natCoo[1]) / 4.,
            (1. - natCoo[0]) * (1. + natCoo[1]) / 4.,
        ], ti.f64)


    @ti.func
    def dshape_dnat(self, natCoo):
        """derivative of shape function with respect to natural coodinate"""
        return ti.Matrix([
            [-(1. - natCoo[1]) / 4., -(1. - natCoo[0]) / 4.],
            [ (1. - natCoo[1]) / 4., -(1. + natCoo[0]) / 4.],
            [ (1. + natCoo[1]) / 4.,  (1. + natCoo[0]) / 4.],
            [-(1. + natCoo[1]) / 4.,  (1. - natCoo[0]) / 4.]
        ], ti.f64)


    def shapeFunc_pyscope(self, natCoo: np.ndarray):  
        """input the natural coordinate and get the shape function values"""
        return np.array([
            (1. - natCoo[0]) * (1. - natCoo[1]) / 4.,
            (1. + natCoo[0]) * (1. - natCoo[1]) / 4.,
            (1. + natCoo[0]) * (1. + natCoo[1]) / 4.,
            (1. - natCoo[0]) * (1. + natCoo[1]) / 4.,
        ])


    def dshape_dnat_pyscope(self, natCoo):
        """derivative of shape function with respect to natural coodinate"""
        return np.array([
            [-(1. - natCoo[1]) / 4., -(1. - natCoo[0]) / 4.],
            [ (1. - natCoo[1]) / 4., -(1. + natCoo[0]) / 4.],
            [ (1. + natCoo[1]) / 4.,  (1. + natCoo[0]) / 4.],
            [-(1. + natCoo[1]) / 4.,  (1. - natCoo[0]) / 4.]
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

        area_x_weight = np.linalg.norm(nodes[facet[0]] - nodes[facet[1]]) * \
                                self.facet_point_weights[facet][integPointId]

        return global_normal, area_x_weight


    @ti.func
    def strain_for_stiffnessMtrx_taichi(self, dsdx):
        """
        strain for the stiffness matrix:
        with shape = (n, m), 
            n is the dimension of strian in Voigt notation, 
                e.g., n = 3 for components including epsilon_11, epsilon_22 and gamma_12 (= 2 * epsilon_12)
            m is the number of dof of this element, 
                e.g., m = 12 for qudritic triangular element
        """
        return ti.Matrix([ 
            [dsdx[0, 0], 0.,
             dsdx[1, 0], 0.,
             dsdx[2, 0], 0.,
             dsdx[3, 0], 0.,],

            [0., dsdx[0, 1],
             0., dsdx[1, 1],
             0., dsdx[2, 1],
             0., dsdx[3, 1],],

            [dsdx[0, 1], dsdx[0, 0],
             dsdx[1, 1], dsdx[1, 0],
             dsdx[2, 1], dsdx[2, 0],
             dsdx[3, 1], dsdx[3, 0],]
        ])


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
                (ele[0], ele[2], ele[3]), 
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

        3-------------------2
        |                   |
        |  (3)         (2)  |
        |                   |
        |                   |
        |                   |
        |  (0)         (1)  |
    η   |                   |
    ^   0-------------------1
    |
    ---> ξ 
        """
        tmp = 3.**0.5
        natCoos = ti.Matrix([  # natural coordinates of outer nodes
            [-tmp, -tmp],
            [ tmp, -tmp],
            [ tmp,  tmp],
            [-tmp,  tmp],
        ])
        for ele in nodal_vals:
            vec = ti.Vector([internal_vals[ele, i] for i in range(self.gaussPoints.shape[0])])
            for node in range(nodal_vals[ele].n):
                nodal_vals[ele][node] = (self.shapeFunc(natCoos[node, :]) * vec).sum()
