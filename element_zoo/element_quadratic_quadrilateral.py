import numpy as np
import taichi as ti
from element_base import ElementBase

"""
        3----6----2
        |         |
        7         5
    η   |         | 
    ^   0----4----1
    |
    ---> ξ 
"""
@ti.data_oriented
class Element_quadratic_quadrilateral(ElementBase):
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
            (0, 4): [[-1., -1.], [0., -1.]], (1, 4): [[1., -1.], [0., -1.]],
            (1, 5): [[1., -1.], [1., 0.]],   (2, 5): [[1., 1.], [1., 0.]],
            (2, 6): [[1., 1.], [0., 1.]],    (3, 6): [[-1., 1.], [0., 1.]],
            (0, 7): [[-1., 1.], [-1., 0.]],  (3, 7): [[-1., -1.], [-1., 0.]] 
        }
        self.facet_point_weights = {
            (0, 4): [0.5, 0.5],   (1, 4): [0.5, 0.5],
            (1, 5): [0.5, 0.5],   (2, 5): [0.5, 0.5],
            (2, 6): [0.5, 0.5],   (3, 6): [0.5, 0.5],
            (0, 7): [0.5, 0.5],   (3, 7): [0.5, 0.5] 
        }
        self.integPointNum_eachFacet = len(list(self.facet_point_weights.values())[0])

        """ facet normals in natural coordinate,
            must points to the outside of the element"""
        self.facet_natural_normals = {  
            (0, 4): [[0., -1.], [0., -1.]],  (1, 4): [[0., -1.], [0., -1.]],
            (1, 5): [[1., 0.], [1., 0.]],  (2, 5): [[1., 0.], [1., 0.]],
            (2, 6): [[0., 1.], [0., 1.]],  (3, 6): [[0., 1.], [0., 1.]],
            (0, 7): [[-1., 0.], [-1., 0.]], (3, 7): [[-1., 0.], [-1., 0.]] 
        }
        """facet number for reading .inp (Abaqus, CalculiX) file and get the face set"""
        self.inp_surface_num = [((0, 4), (1, 4)), 
                                ((1, 5), (2, 5)),  
                                ((2, 6), (3, 6)),  
                                ((0, 7), (3, 7))]  


    @ti.func
    def shapeFunc(self, nc):  # nc stands for natrual coordinates
        """input the natural coordinate and get the shape function values"""
        return ti.Vector([
            (1. - nc[0]) * (1. - nc[1]) * (-1. - nc[0] - nc[1]) / 4.,
            (1. + nc[0]) * (1. - nc[1]) * (-1. + nc[0] - nc[1]) / 4.,
            (1. + nc[0]) * (1. + nc[1]) * (-1. + nc[0] + nc[1]) / 4.,
            (1. - nc[0]) * (1. + nc[1]) * (-1. - nc[0] + nc[1]) / 4.,

            (1. - nc[0]**2) * (1. - nc[1]) / 2.,
            (1. - nc[1]**2) * (1. + nc[0]) / 2.,
            (1. - nc[0]**2) * (1. + nc[1]) / 2.,
            (1. - nc[1]**2) * (1. - nc[0]) / 2.,
        ], ti.f64)


    @ti.func
    def dshape_dnat(self, nc):  # nc stands for natrual coordinates
        """derivative of shape function with respect to natural coodinate"""
        return ti.Matrix([
            [-(1. - nc[1]) * (-2. * nc[0] - nc[1]) / 4., 
             -(1. - nc[0]) * (-2. * nc[1] - nc[0]) / 4.],
            
            [ (1. - nc[1]) * ( 2. * nc[0] - nc[1]) / 4., 
             -(1. + nc[0]) * (-2. * nc[1] + nc[0]) / 4.],

            [ (1. + nc[1]) * ( 2. * nc[0] + nc[1]) / 4., 
              (1. + nc[0]) * ( 2. * nc[1] + nc[0]) / 4.],
            
            [-(1. + nc[1]) * (-2. * nc[0] + nc[1]) / 4., 
              (1. - nc[0]) * ( 2. * nc[1] - nc[0]) / 4.],

            [-2.*nc[0] * (1. - nc[1]) / 2., 
             -(1. - nc[0]**2) / 2.],

            [(1. - nc[1]**2) / 2., 
             -2.*nc[1] * (1. + nc[0]) / 2.],

            [-2.*nc[0] * (1. + nc[1]) / 2., 
             (1. - nc[0]**2) / 2.],

            [-(1. - nc[1]**2) / 2., 
             -2.*nc[1] * (1. - nc[0]) / 2.],
        ], ti.f64)


    def shapeFunc_pyscope(self, nc: np.ndarray):  
        """input the natural coordinate and get the shape function values"""
        return np.array([
            (1. - nc[0]) * (1. - nc[1]) * (-1. - nc[0] - nc[1]) / 4.,
            (1. + nc[0]) * (1. - nc[1]) * (-1. + nc[0] - nc[1]) / 4.,
            (1. + nc[0]) * (1. + nc[1]) * (-1. + nc[0] + nc[1]) / 4.,
            (1. - nc[0]) * (1. + nc[1]) * (-1. - nc[0] + nc[1]) / 4.,

            (1. - nc[0]**2) * (1. - nc[1]) / 2.,
            (1. - nc[1]**2) * (1. + nc[0]) / 2.,
            (1. - nc[0]**2) * (1. + nc[1]) / 2.,
            (1. - nc[1]**2) * (1. - nc[0]) / 2.,
        ])


    def dshape_dnat_pyscope(self, nc: np.ndarray):
        """derivative of shape function with respect to natural coodinate"""
        return np.array([
            [-(1. - nc[1]) * (-2. * nc[0] - nc[1]) / 4., 
             -(1. - nc[0]) * (-2. * nc[1] - nc[0]) / 4.],
            
            [ (1. - nc[1]) * ( 2. * nc[0] - nc[1]) / 4., 
             -(1. + nc[0]) * (-2. * nc[1] + nc[0]) / 4.],

            [ (1. + nc[1]) * ( 2. * nc[0] + nc[1]) / 4., 
              (1. + nc[0]) * ( 2. * nc[1] + nc[0]) / 4.],
            
            [-(1. + nc[1]) * (-2. * nc[0] + nc[1]) / 4., 
              (1. - nc[0]) * ( 2. * nc[1] - nc[0]) / 4.],

            [-2.*nc[0] * (1. - nc[1]) / 2., 
             -(1. - nc[0]**2) / 2.],

            [(1. - nc[1]**2) / 2., 
             -2.*nc[1] * (1. + nc[0]) / 2.],

            [-2.*nc[0] * (1. + nc[1]) / 2., 
             (1. - nc[0]**2) / 2.],

            [-(1. - nc[1]**2) / 2., 
             -2.*nc[1] * (1. - nc[0]) / 2.],
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

        area_x_weight = np.linalg.norm(nodes[facet[0]] - nodes[facet[1]]) * \
                                self.facet_point_weights[facet][integPointId]

        return global_normal, area_x_weight


    @ti.func
    def strainMtrx(self, dsdx):
        """
        strain for the stiffness matrix:
        with shape = (n, m), 
            n is the dimension of strian in Voigt notation, 
                e.g., n = 3 for components including epsilon_11, epsilon_22 and gamma_12 (= 2 * epsilon_12)
            m is the number of dof of this element, 
                e.g., m = 12 for qudritic triangular element
        """
        return ti.Matrix([ 
            [dsdx[0, 0], 0.,    dsdx[1, 0], 0.,
             dsdx[2, 0], 0.,    dsdx[3, 0], 0.,
             dsdx[4, 0], 0.,    dsdx[5, 0], 0.,
             dsdx[6, 0], 0.,    dsdx[7, 0], 0.,],

            [0., dsdx[0, 1],    0., dsdx[1, 1],
             0., dsdx[2, 1],    0., dsdx[3, 1],
             0., dsdx[4, 1],    0., dsdx[5, 1],
             0., dsdx[6, 1],    0., dsdx[7, 1],],

            [dsdx[0, 1], dsdx[0, 0],    dsdx[1, 1], dsdx[1, 0],
             dsdx[2, 1], dsdx[2, 0],    dsdx[3, 1], dsdx[3, 0],
             dsdx[4, 1], dsdx[4, 0],    dsdx[5, 1], dsdx[5, 0],
             dsdx[6, 1], dsdx[6, 0],    dsdx[7, 1], dsdx[7, 0],]
        ])


    def getMesh(self, elements: np.ndarray):
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
                (ele[0], ele[4], ele[7]), (ele[1], ele[4], ele[5]),
                (ele[2], ele[5], ele[6]), (ele[3], ele[6], ele[7]), 
                (ele[5], ele[6], ele[7]), (ele[4], ele[5], ele[7])
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


    @ti.func
    def shapeFunc_for_extrapolate(self, natCoo):  
        """input the natural coordinate and get the shape function values"""
        return ti.Vector([
            (1. - natCoo[0]) * (1. - natCoo[1]) / 4.,
            (1. + natCoo[0]) * (1. - natCoo[1]) / 4.,
            (1. + natCoo[0]) * (1. + natCoo[1]) / 4.,
            (1. - natCoo[0]) * (1. + natCoo[1]) / 4.,
        ], ti.f64)


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

        3---------6---------2
        |                   |
        |  (3)         (2)  |
        |                   |
        7                   5
        |                   |
        |  (0)         (1)  |
    η   |                   |
    ^   0---------4---------1
    |
    ---> ξ 
        """
        tmp = 3.**0.5
        natCoos = ti.Matrix([  # natural coordinates of outer nodes
            [-tmp, -tmp],
            [ tmp, -tmp],
            [ tmp,  tmp],
            [-tmp,  tmp],
            [0., -tmp],
            [tmp, 0.],
            [0., tmp],
            [-tmp, 0.]
        ])
        for ele in nodal_vals:
            vec = ti.Vector([internal_vals[ele, i] for i in range(internal_vals.shape[1])])
            for node in range(nodal_vals[ele].n):
                nodal_vals[ele][node] = (self.shapeFunc_for_extrapolate(natCoos[node, :]) * vec).sum()


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    vecs = np.array([1., 1., 1, 1, 1, 1, 1, 1])
    # vecs = np.array([1., 0., 0, 0, 0, 0, 0, 0])
    # vecs = np.array([1., 1., 0, 0, 1, 0, 0, 0])
    ELE = Element_quadratic_quadrilateral()
    a = ELE.shapeFunc_pyscope(np.array([-0.5, -0.5])) @ vecs
    b = ELE.shapeFunc_pyscope(np.array([0.5, -0.5])) @ vecs
    c = ELE.shapeFunc_pyscope(np.array([0.5, 0.5])) @ vecs
    d = ELE.shapeFunc_pyscope(np.array([-0.5, 0.5])) @ vecs
    e = ELE.shapeFunc_pyscope(np.array([1., 0.3])) @ vecs
    print("a, b, c, d, e = {}, {}, {}, {}, {}".format(a, b, c, d, e))
    print("ELE.shapeFunc_pyscope(np.array([-0.5, -0.5])) = ", ELE.shapeFunc_pyscope(np.array([-0.5, -0.5])))
    print("sum(ELE.shapeFunc_pyscope(np.array([-0.5, -0.5]))) = ", sum(ELE.shapeFunc_pyscope(np.array([-0.5, -0.5]))))
    print("ELE.shapeFunc_pyscope(np.array([1., 0.3])) = ", ELE.shapeFunc_pyscope(np.array([1., 0.3])))
    print("sum(ELE.shapeFunc_pyscope(np.array([1., 0.3]))) = ", sum(ELE.shapeFunc_pyscope(np.array([1., 0.3]))))

    grad1 = vecs @ ELE.dshape_dnat_pyscope(np.array([-0.5, -1.]))
    grad2 = vecs @ ELE.dshape_dnat_pyscope(np.array([0.5, -1.]))
    grad3 = vecs @ ELE.dshape_dnat_pyscope(np.array([0.5, 0.5]))
    grad4 = vecs @ ELE.dshape_dnat_pyscope(np.array([-0.5, 0.5]))
    print("grad1, grad2, grad3, grad4 = {}, {}, {}, {}".format(grad1, grad2, grad3, grad4))