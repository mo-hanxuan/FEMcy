"""
construct the stiffness matrix
"""
import taichi as ti
import numpy as np
import time
from body import Body
from readInp import *
from material import *
from linear_triangular_element import Linear_triangular_element
from quadritic_triangular_element import Quadritic_triangular_element
from math_library import relative_error
from conjugateGradientSolver import ConjugateGradientSolver_rowMajor as CG
from tiMath import a_equals_b_plus_c_mul_d, a_from_b, c_equals_a_minus_b, field_abs_max


@ti.func
def vec_mul_voigtMtrx_2d(vec, mtrx):  # vector multiply voigt matrix
    """
    dot product of vector and voigt matrix
    for 2-dimension
        matrix = [
            [mtrx[0], mtrx[2]],
            [mtrx[2], mtrx[1]]
        ]
        thus, vec * matrix = [
            vec[0] * mtrx[0, :] + vec[1] * mtrx[2, :], 
            vec[0] * mtrx[2, :] + vec[1] * mtrx[1, :], 
        ]
    """
    return [
        vec[0] * mtrx[0, :] + vec[1] * mtrx[2, :], 
        vec[0] * mtrx[2, :] + vec[1] * mtrx[1, :], 
    ]


@ti.data_oriented
class System_of_equations:
    """
    the system of equations that applies to solve the linear equation systems (can be modified to nonlinear system)
    including:
        sparseMtrx: the sparse stiffness matrix utilized to solve the dofs
        rhs: right hand side of the equation system
    """
    def __init__(self, body: Body, material):
        self.dm = body.dm  # spatial dimension of the system
        self.body = body
        self.elements, self.nodes = body.elements, body.nodes 
        
        self.sparseMtrx = ti.field(ti.f64)
        sparseMtrx_components = ti.root.pointer(ti.ij, (body.nodes.shape[0] * body.dm, body.nodes.shape[0] * body.dm))
        sparseMtrx_components.place(self.sparseMtrx)
        
        ### sparseMtrx @ dof = rhs
        self.rhs = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # right hand side of the equation system
        self.dof = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # degree of freedom that needs to be solved
        self.rhs_before_dirichlet = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))

        ### define the element types of the body, must be modified latter!!!
        if body.np_elements.shape[1] == 3:
            self.ELE = Linear_triangular_element()
        elif body.np_elements.shape[1] == 6:
            self.ELE = Quadritic_triangular_element()

        ### variables related to geometric nonlinear
        self.body.cauchy_stress = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))                                
        self.nodal_force = ti.field(ti.f64, shape=(self.dm * self.body.nodes.shape[0]))  
        self.residual_nodal_force = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))

        ### dnatdx of each integration point of each element
        self.dnatdxs = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                                       shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))

        ### dsdx (derivative of shape functio with respect to current coordinate), and volume of each guass point
        self.dsdx = ti.Matrix.field(self.elements[0].n, self.dm, ti.f64, 
                                    shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        self.vol = ti.field(ti.f64, shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        
        ### constitutive material (e.g., elastic constants) of each element
        self.material = material; self.C = material.ti_C
        self.ddsdde = ti.Matrix.field(n=material.C.shape[0], m=material.C.shape[1], dtype=ti.f64, 
                                      shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        self.ddsdde_init()
        
        ### link a node to the related elements
        body.get_nodeEles()
        maxLen = max(len(eles) for eles in body.nodeEles)
        self.nodeEles = ti.Vector.field(maxLen, ti.i32, shape=(self.nodes.shape[0], ))
        nodeEles = -np.ones((body.nodes.shape[0], maxLen))
        for node in range(len(nodeEles)):
            nodeEles[node, 0:len(body.nodeEles[node])] = body.nodeEles[node][:]
        self.nodeEles.from_numpy(nodeEles)

        ### get the sparseIJ
        body.get_coElement_nodes()
        maxLen = max(len(body.coElement_nodes[node]) for node in range(body.nodes.shape[0]))
        self.sparseIJ = ti.Vector.field(maxLen * self.dm, ti.i32, shape=(self.nodes.shape[0] * self.dm, ))
        sparseIJ = -np.ones((self.nodes.shape[0] * self.dm, maxLen * self.dm))
        for node0 in range(self.nodes.shape[0]):
            js = [node1 * self.dm + i for node1 in body.coElement_nodes[node0] for i in range(self.dm)]
            for i in range(self.dm):
                sparseIJ[node0 * self.dm + i][:len(js)] = js[:]
        self.sparseIJ.from_numpy(sparseIJ)

        ### init the row major form of sparse matrix
        print("\033[32;1m shape of the sparseIJ is {} \033[0m".format(self.sparseIJ.shape))
        self.sparseMtrx_rowMajor = ti.Vector.field(self.sparseIJ[0].n, ti.f64, shape=(self.sparseIJ.shape[0], ))


    @ti.kernel
    def ddsdde_init(self, ):
        """get the ddsdde at each integration point of each element"""
        for ele, igp in self.ddsdde:
            self.ddsdde[ele, igp] = self.C


    @ti.kernel
    def get_dnatdxs(self, ):
        ### compute dNatdx (derivative of natural coordinates with respect to x) of all integration points
        for ele in self.elements:
            localNodes = ti.Matrix([[0. for i in range(self.dm)] for j in range(self.elements[0].n)], ti.f64)
            for i in range(localNodes.n):
                for j in range(localNodes.m):
                    localNodes[i, j] = self.nodes[self.elements[ele][i]][j]
            for id in range(self.ELE.gaussPoints.shape[0]):
                natCoo = self.ELE.gaussPoints[id]
                dsdn = self.ELE.dshape_dnat(natCoo)
                dxdn = ti.Matrix([[0. for i in range(self.dm)] for j in range(self.dm)], ti.f64)
                dxdn = localNodes.transpose() @ dsdn  # sum over the shape id
                self.dnatdxs[ele, id] = dxdn.inverse()


    @ti.kernel
    def assemble_sparseMtrx_old(self, ):
        dm = self.body.dm
        ### refresh the sparse Matrix if it has been assemble in the previous increment
        for i, j in self.sparseMtrx:
            self.sparseMtrx[i, j] = 0.
        for node0 in self.nodes:
            for iele in range(self.nodeEles[0].n):
                if self.nodeEles[node0][iele] != -1:
                    ele = self.nodeEles[node0][iele]
                    
                    localNodes = ti.Matrix([[0. for i in range(self.dm)] for j in range(self.elements[0].n)], ti.f64)
                    for i in range(localNodes.n):
                        for j in range(localNodes.m):
                            localNodes[i, j] = self.nodes[self.elements[ele][i]][j]

                    ### get the gradient of u, express it as the coefficients of u of different nodes
                    for igp in range(self.ELE.gaussPoints.shape[0]):
                        gp = self.ELE.gaussPoints[igp]
                        dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                        dsdx = dsdn @ self.dnatdxs[ele, igp]  # dshape / dx
                        
                        ### length of each component = dm * nodes_of_element
                        strain = self.ELE.strain_for_stiffnessMtrx_taichi(dsdx)
                        ### C : ε， each component is a vector with dimension s, thus maybe can not use operation @ directly
                        stress_voigt = self.ddsdde[ele, igp] @ strain

                        ### get the sequence of this node in the element
                        nid = 0  # nid = list(body.np_elements[ele, :]).index(node0)
                        for i in range(self.elements[0].n):
                            if self.elements[ele][i] == node0:
                                nid = i
                        ### modified latter to automatically adjust to 2d and 3d
                        dsdx_x_stress = vec_mul_voigtMtrx_2d(dsdx[nid, :], stress_voigt)  
                        
                        ### get the volume related to this Gauss point
                        vol = (localNodes.transpose() @ dsdn).determinant() * self.ELE.gaussWeights[igp]

                        ### integrate to the large sparse matrix
                        Is = ti.Vector([node0 * dm + 0, 
                                        node0 * dm + 1])
                        Js_ = self.elements[ele] * dm  ## need to be modified here
                        Js = ti.Vector([x + i for x in Js_ for i in range(2)])
                        for i_local in ti.static(range(Is.n)):
                            i_global = Is[i_local]
                            for j_local in ti.static(range(Js.n)):
                                j_global = Js[j_local]
                                ### don't use +=, the atomic add will slow the speed
                                self.sparseMtrx[i_global, j_global] = \
                                self.sparseMtrx[i_global, j_global] + dsdx_x_stress[i_local][j_local] * vol
        ### transform the sparse matrix into row major
        for i in self.sparseMtrx_rowMajor:
            for j in range(self.sparseMtrx_rowMajor[0].n):
                self.sparseMtrx_rowMajor[i][j] = 0.
        for i, j in self.sparseMtrx:
            j0 = self.sparseMatrix_get_j(i, j)
            self.sparseMtrx_rowMajor[i][j0] = self.sparseMtrx[i, j]
    

    @ti.kernel
    def assemble_sparseMtrx(self, ):
        dm, sparseMtrx, sparseMtrx_rowMajor, \
        nodeEles, nodes, elements, ddsdde  = ti.static(
            self.body.dm, self.sparseMtrx, self.sparseMtrx_rowMajor, \
            self.nodeEles, self.nodes, self.elements, self.ddsdde)
        ### refresh the sparse Matrix if it has been assemble in the previous increment
        for i, j in sparseMtrx:
            sparseMtrx[i, j] = 0.
        for node0 in nodes:
            for iele in range(nodeEles[0].n):
                if nodeEles[node0][iele] != -1:
                    ele = nodeEles[node0][iele]

                    ### get the sequence of this node in the element
                    nid = 0  # nid = list(body.np_elements[ele, :]).index(node0)
                    for i in range(elements[0].n):
                        if elements[ele][i] == node0:
                            nid = i

                    ### get the gradient of u, express it as the coefficients of u of different nodes
                    for igp in range(self.ELE.gaussPoints.shape[0]):
                        dsdx = self.dsdx[ele, igp]
                        
                        ### length of each component = dm * nodes_of_element
                        strain = self.ELE.strain_for_stiffnessMtrx_taichi(dsdx)
                        ### C : ε， each component is a vector with dimension s, thus maybe can not use operation @ directly
                        stress_voigt = ddsdde[ele, igp] @ strain

                        ### modified latter to automatically adjust to 2d and 3d
                        dsdx_x_stress = vec_mul_voigtMtrx_2d(dsdx[nid, :], stress_voigt)  
                        
                        ### get the volume related to this Gauss point
                        vol = self.vol[ele, igp]

                        ### integrate to the large sparse matrix
                        Is = ti.Vector([node0 * dm + 0, 
                                        node0 * dm + 1])
                        Js_ = elements[ele] * dm  ## need to be modified here
                        Js = ti.Vector([x + i for x in Js_ for i in range(2)])
                        for i_local in ti.static(range(Is.n)):
                            i_global = Is[i_local]
                            for j_local in ti.static(range(Js.n)):
                                j_global = Js[j_local]
                                ### don't use +=, the atomic add will slow the speed
                                sparseMtrx[i_global, j_global] = \
                                sparseMtrx[i_global, j_global] + dsdx_x_stress[i_local][j_local] * vol
        ### transform the sparse matrix into row major
        for i in sparseMtrx_rowMajor:
            for j in range(sparseMtrx_rowMajor[0].n):
                sparseMtrx_rowMajor[i][j] = 0.
        for i, j in sparseMtrx:
            j0 = self.sparseMatrix_get_j(i, j)
            sparseMtrx_rowMajor[i][j0] = sparseMtrx[i, j]
  

    @ti.kernel
    def dirichletBC_linearEquations(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    ):
        """apply dirichlet boundary condition to the body
           modify the sparse matrix and the rhs (right hand side)
           this is for linear equation systems"""
        ### impose dirichlet BC at specific nodes
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified

            ### modify the right hand side
            for j0 in range(self.sparseIJ[0].n):
                j_global = self.sparseIJ[i_global][j0]
                if j_global != -1:
                    ### use the symmetric property of the sparse matrix, find sparseMtrx[j_global, i_global]
                    i0 = self.sparseMatrix_get_j(j_global, i_global)
                    self.rhs[j_global] = self.rhs[j_global] - sval * self.sparseMtrx_rowMajor[j_global][i0]
            self.rhs[i_global] = sval

            ### modify the sparse matrix
            for j0 in range(self.sparseIJ[0].n):
                j_global = self.sparseIJ[i_global][j0]
                if j_global != -1:
                    self.sparseMtrx_rowMajor[i_global][j0] = 0.
                    i0 = self.sparseMatrix_get_j(j_global, i_global)
                    self.sparseMtrx_rowMajor[j_global][i0] = 0.
            i0 = self.sparseMatrix_get_j(i_global, i_global)
            self.sparseMtrx_rowMajor[i_global][i0] = 1.


    def dirichletBC_forNewtonMethod(self, inp: Inp_info):
        dirichletBCs = inp.dirichlet_bc_info
        for dirichletBC in dirichletBCs:
            self.dirichletBC_forNewtonMethod_kernel(nodeSet=dirichletBC["node_set"], 
                                                    dm_specified=dirichletBC["dof"],
                                                    sval=dirichletBC["val"])
    @ti.kernel
    def dirichletBC_forNewtonMethod_kernel(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    ):
        """apply dirichlet boundary condition to the body
           modify the sparse matrix and the residual force
           this is for Newton method, ref: https://scorec.rpi.edu/~granzb/notes/dbcs/dbcs.pdf """
        
        ## impose dirichlet BC at dof
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified
            self.dof[i_global] = sval
        
        ### impose dirichlet BC at residual force and sparse matrix
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified

            ### modify the residual force
            self.residual_nodal_force[i_global] = 0.

            ### modify the sparse matrix (i.e., the Jacobian)
            for j0 in range(self.sparseIJ[0].n):
                j_global = self.sparseIJ[i_global][j0]
                if j_global != -1:
                    self.sparseMtrx_rowMajor[i_global][j0] = 0.
                    i0 = self.sparseMatrix_get_j(j_global, i_global)
                    self.sparseMtrx_rowMajor[j_global][i0] = 0.
            i0 = self.sparseMatrix_get_j(i_global, i_global)
            self.sparseMtrx_rowMajor[i_global][i0] = 1.
    

    @ti.kernel
    def dirichletBC_dof(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    ):
        ### impose dirichlet BC at dof
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified
            self.dof[i_global] = sval


    def neumannBC(self, load_facets, load_val: float, load_dir=np.array([])):  # Neumann boundary condition, 
                                                                       # should be modified latter to taichi version!!!
        """
        load_facets: list(tuple), the boundary facets with load (i.e., the boundary with specific flux value)
        freeload_facets: boundary without specified value of flux(field gradient)

            if no load in the surface, flux term doesn't need to be computed, because:
                1.  free surface, with no traction force on the surface, thus the flux term is 0
                2.  surface with Dirichlet BC, nodal displacement is specified, 
                    flux term is not 0, but can be automatically deduced from Dirichlet BC
                    (technically, this term doesn't need to be computed, 
                    because after zero-one setting method for DirichletBC, the effort here will be erased) 
        """
        body = self.body; ELE = self.ELE
        body.get_boundary()
        load_nodes = set()  # node set of all load facets
        for facet in load_facets:
            load_nodes |= {*facet}
        
        for node0 in load_nodes:
            for facet in body.node2boundary[node0]:
                if facet in load_facets:  
                    ele = body.boundary[facet]

                    ### obtain the facet normals on the free-load boundary 
                    ###     (points to the outside of the element)
                    localNodes = np.array([body.np_nodes[node, :] for node in body.np_elements[ele, :]])
                    eleNodesList = body.np_elements[ele, :].tolist()
                    localFacet = [eleNodesList.index(i) for i in facet]
                    for gaussId in range(ELE.gaussPointNum_eachFacet):
                        normal_vector, area_x_gaussWeight = ELE.global_normal(nodes=localNodes, 
                                                                                facet=localFacet, 
                                                                                gaussPointId=gaussId)
                        ### get the flux, which can also be interpreted as traction force
                        if len(load_dir) == 0: 
                            flux = load_val * normal_vector * area_x_gaussWeight
                        else:
                            flux = load_val * load_dir * area_x_gaussWeight
                        
                        ### get the sequence of this node in the element
                        natCoo = ELE.facet_natural_coos[tuple(sorted(localFacet))][gaussId]
                        nid = list(body.np_elements[ele, :]).index(node0)
                        shapeVal = ELE.shapeFunc_pyscope(natCoo)[nid]  # dshape/dx, you need the sequence of this node in the element
                        
                        for i in range(self.dm):
                            self.rhs[node0*self.dm + i] = self.rhs[node0*self.dm + i] + flux[i] * shapeVal

    
    @ti.func
    def sparseMatrix_get_j(self, i_global, j_global):
        j_local = 0
        for j in range(self.sparseIJ[0].n):
            if self.sparseIJ[i_global][j] == j_global:
                j_local = j
        return j_local
        

    @ti.kernel
    def count_components_of_sparseMtrx(self, ) -> float:
        count = 0
        for i, j in self.sparseMtrx:
            count += 1
        return count
    

    def check_sparseIJ(self, ):
        """check whether indexes repeatly appear in sparseIJ"""
        for i in range(self.sparseIJ.shape[0]):
            js = set()
            for j0 in range(self.sparseIJ[0].n):
                j = self.sparseIJ[i][j0]
                if j != -1:
                    if j in js:
                        print("\033[31;1m error, {}, {} appear repeatly \033[0m".format(i, j))
                    else:
                        js.add(j)
    

    def solve_dof(self, geometric_nonlinear: bool=False):
        if not geometric_nonlinear:
            solver = CG(spm=self.sparseMtrx_rowMajor, sparseIJ=self.sparseIJ, b=self.rhs)
        else:
            solver = CG(spm=self.sparseMtrx_rowMajor, sparseIJ=self.sparseIJ, b=self.residual_nodal_force)
        solver.solve()

        if not geometric_nonlinear:
            self.dof = solver.x
        else:
            ### self.dof = self.dof - solver.x
            c_equals_a_minus_b(self.dof, self.dof, solver.x)
        
        return solver


    def compute_strain_stress(self, 
                            for_visualize=True,  # define whether this operation is for visualization
                            ):
        dm = self.dm
        if for_visualize:
            gaussPoints = self.ELE.gaussPoints_visualize
        else:
            gaussPoints = self.ELE.gaussPoints
        ### strain: each gauss point in each element has a strain, use Voigt notation
        self.body.strains = ti.Vector.field(dm + dm*(dm-1)//2, ti.f64, 
                                            shape=(self.elements.shape[0], gaussPoints.shape[0]))
        self.body.stresses = ti.Vector.field(dm + dm*(dm-1)//2, ti.f64, 
                                             shape=(self.elements.shape[0], gaussPoints.shape[0]))
        self.body.mises_stresses = ti.field(ti.f64, 
                                            shape=(self.elements.shape[0], gaussPoints.shape[0]))
        
        self.compute_strain_stress_kernel(gaussPoints)
        return self.body.strains, self.body.stresses
    

    @ti.kernel
    def compute_strain_stress_kernel(self, gaussPoints: ti.template()):
        body = self.body
        dm, elements, nodes, ddsdde = ti.static(
            self.dm, self.elements, self.nodes, self.ddsdde
        )
        for ele in elements:
            local_us = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            local_nodes = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            for node_ in range(elements[0].n):
                node = elements[ele][node_]
                for j in ti.static(range(dm)):
                    local_us[node_, j] = self.dof[node * dm + j]  # self.u[node][j]
                    local_nodes[node_, j] = nodes[node][j]
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdx = (local_nodes.transpose() @ dsdn).inverse()  # !!!!!!!! look at here, this should referenced at the current configuration!!!
                dsdx = dsdn @ dnatdx  # strain = (u_node1, u_node2, u_node3) * dsdx
                dudx = local_us.transpose() @ dsdx
                
                body.strains[ele, igp][0] = dudx[0, 0]
                body.strains[ele, igp][1] = dudx[1, 1]
                body.strains[ele, igp][2] = dudx[0, 1] + dudx[1, 0]

                if igp < ddsdde.shape[1]:
                    """body.stresses[ele, igp] = ddsdde[ele, igp] @ body.strains[ele, igp] 
                       but I don't know why sometimes operation '@' goes wrong here"""
                    for i in range(dm):
                        body.stresses[ele, igp][i] = 0.
                        for j in range(dm):
                            body.stresses[ele, igp][i] = \
                                body.stresses[ele, igp][i] + \
                                    ddsdde[ele, igp][i, j] * body.strains[ele, igp][j]
                
                else:  # visualized Gauss points > original Gauss Points
                    C = ti.Matrix(  # C is the average elastic tensor of original Gauss Points
                        [[0. for _ in range(ddsdde[ele, 0].m)] for _ in range(ddsdde[ele, 0].n)])
                    for i in range(ddsdde.shape[1]):
                        C = C + ddsdde[ele, i]
                    C /= ddsdde.shape[1]
                    body.stresses[ele, igp] = C @ body.strains[ele, igp]


    def impose_boundary_condition(self, inp: Inp_info, geometric_nonlinear: bool=False):
        ### =========== apply the boundary condition ===========
        ### first, apply Neumann BC
        neumannBCs = inp.neumann_bc_info
        for neumannBC in neumannBCs:
            if "direction" in neumannBC:
                self.neumannBC(neumannBC["face_set"], 
                                load_val=neumannBC["traction"], 
                                load_dir=neumannBC["direction"])
            else:
                self.neumannBC(neumannBC["face_set"], 
                                load_val=neumannBC["traction"])
        a_from_b(self.rhs_before_dirichlet, self.rhs)
        ### then, apply Dirichlet BC
        dirichletBCs = inp.dirichlet_bc_info
        if geometric_nonlinear == False:
            for dirichletBC in dirichletBCs:
                node_set = ti.field(ti.i32, shape=(len(dirichletBC["node_set"])))
                node_set.from_numpy(np.array([*dirichletBC["node_set"]]))
                dirichletBC["node_set"] = node_set  # replace the original node set by field
                self.dirichletBC_linearEquations(
                        node_set, dirichletBC["dof"], 
                        dirichletBC["val"])
        else:  # large deformation (geometric nonlinear), dirichlet BC is imposed latter at Newton's method
            for dirichletBC in dirichletBCs:
                node_set = ti.field(ti.i32, shape=(len(dirichletBC["node_set"])))
                node_set.from_numpy(np.array([*dirichletBC["node_set"]]))
                dirichletBC["node_set"] = node_set  # replace the original node set by field
                self.dirichletBC_dof(
                        node_set, dirichletBC["dof"], 
                        dirichletBC["val"])


    @ti.kernel
    def plane_strain_nonlinear_constitutive(self, gaussPoints: ti.template()):
        """geometric nonlinear constitutive of plane strain
           get the stress of each integration point"""
        body = self.body
        dm, elements, nodes, ddsdde = ti.static(
            self.dm, self.elements, self.nodes, self.ddsdde)
        eye = ti.Matrix([ 
            [1., 0.],
            [0., 1.],
        ])
        for ele in elements:
            local_us = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            local_nodes = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            for node_ in range(elements[0].n):
                node = elements[ele][node_]
                for j in ti.static(range(dm)):
                    local_us[node_, j] = self.dof[node * dm + j]  # self.u[node][j]
                    local_nodes[node_, j] = nodes[node][j]
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdX = (local_nodes.transpose() @ dsdn).inverse()  # deformation gradient refers to initial configuration
                dsdX = dsdn @ dnatdX  # strain = (u_node1, u_node2, u_node3) * dsdx
                dudX = local_us.transpose() @ dsdX
                
                ### get the deformation gradient, F = I + dudX
                F = ti.Matrix([[0. for _ in range(dudX.m)] for _ in range(dudX.n)])
                F = dudX + eye
                
                ### get the Green Strain, E
                E = (F.transpose() @ F - eye) / 2.

                ### get the PK2 stress
                pk2_voigt = ddsdde[ele, igp] @ ti.Vector([E[0, 0], E[1, 1], 
                                                          E[0, 1] + E[1, 0]])
                pk2 = ti.Matrix([ 
                    [pk2_voigt[0], pk2_voigt[2]],
                    [pk2_voigt[2], pk2_voigt[1]]
                ])

                ### get the Cauchy stress
                body.cauchy_stress[ele, igp] = F.determinant()**(-1) * (F @ pk2 @ F.transpose())
    

    @ti.kernel
    def plane_stress_nonlinear_constitutive(self, gaussPoints: ti.template()):
        """geometric nonlinear constitutive of plane stress
           get the stress of each integration point
           constitutive model of plane stress can 
           refer to https://www.comsol.com/blogs/what-is-the-difference-between-plane-stress-and-plane-strain """
        body = self.body
        dm, elements, nodes, nu = ti.static(
            self.dm, self.elements, self.nodes, self.material.poisson_ratio)
        eye_2d = ti.Matrix([ 
            [1., 0.],
            [0., 1.],
        ])
        eye_3d = ti.Matrix([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        for ele in elements:
            local_us = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            local_nodes = ti.Matrix([[0. for _ in range(dm)] for _ in range(elements[0].n)], ti.f64)
            for node_ in range(elements[0].n):
                node = elements[ele][node_]
                for j in ti.static(range(dm)):
                    local_us[node_, j] = self.dof[node * dm + j]  # self.u[node][j]
                    local_nodes[node_, j] = nodes[node][j]
            
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdX = (local_nodes.transpose() @ dsdn).inverse()  # deformation gradient refers to initial configuration
                dsdX = dsdn @ dnatdX  # strain = (u_node1, u_node2, u_node3) * dsdx
                dudX = local_us.transpose() @ dsdX

                ### get the deformation gradient, F = I + dudX
                F = ti.Matrix([[0. for _ in range(dudX.m)] for _ in range(dudX.n)])
                F = dudX + eye_2d

                ### get the deformation gradient at 3d
                F_3d = ti.Matrix([[0. for _ in range(3)] for _ in range(3)])
                F_3d[0:2, 0:2] = F[0:2, 0:2]
                F_3d[2, 2] = -nu / (1. - nu) * (F[0, 0] + F[1, 1] - 2.) + 1.  # deformation at z coordinate

                ### get the Green Strain, E
                E = (F_3d.transpose() @ F_3d - eye_3d) / 2.

                ### get the PK2 stress, voigt notation has been used here, 
                ### modified later by different C at different gauss point
                pk2_voigt = self.material.ti_C_6x6 @ ti.Vector([E[0, 0], E[1, 1], E[2, 2],
                                                                2. * E[0, 1], 2. * E[2, 0], 2. * E[1, 2]])
                pk2 = ti.Matrix([ 
                    [pk2_voigt[0], pk2_voigt[3], pk2_voigt[4]],
                    [pk2_voigt[3], pk2_voigt[1], pk2_voigt[5]],
                    [pk2_voigt[4], pk2_voigt[5], pk2_voigt[2]]
                ])

                ### get the cauchy stress
                stress = F_3d.determinant()**(-1) * (F_3d @ pk2 @ F_3d.transpose())
                # stress = F.determinant()**(-1) * (F @ pk2 @ F.transpose())
                body.cauchy_stress[ele, igp][0:2, 0:2] = stress[0:2, 0:2]


    def assemble_nodal_force_GN(self, ):
        """assemble the nodal force for GN (geometric nonlinear)"""
        ### get all stresses at integration points
        if isinstance(self.material, Linear_isotropic_planeStrain):
            self.plane_strain_nonlinear_constitutive(self.ELE.gaussPoints)  # !!!!!!!!!!!!!!! must be modified latter!!! you should automatically change the constitutive
        elif isinstance(self.material, Linear_isotropic_planeStress):
            self.plane_stress_nonlinear_constitutive(self.ELE.gaussPoints)
        else:
            print("\033[31;1m error! currently we only support these types of materials: "
                  "plane strain and plane stress \033[0m")
        ### get dsdx and vol
        self.get_dsdx_and_vol()
        ### assemble to nodal force
        self.assemble_nodal_force_GN_kernel()


    @ti.kernel 
    def get_dsdx_and_vol(self, ):
        dm, elements, nodes, dof, gaussPoints  = ti.static(
            self.dm, self.elements, self.nodes, self.dof, self.ELE.gaussPoints)
        for ele in self.elements:
            localNodes = ti.Matrix([[0. for i in range(dm)] for j in range(elements[0].n)], ti.f64)
            for i in range(localNodes.n):
                for j in range(localNodes.m):
                    localNodes[i, j] = nodes[elements[ele][i]][j] + \
                                        dof[elements[ele][i] * dm + j]  # nodes coos at current configuration
            
            ### get dsdx and vol of each integration point
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdx = (localNodes.transpose() @ dsdn).inverse()  # at current configuration
                self.dsdx[ele, igp] = dsdn @ dnatdx  # dshape / dx, gradient at current configuration
                self.vol[ele, igp] = (localNodes.transpose() @ dsdn).determinant() * self.ELE.gaussWeights[igp]

    
    @ti.kernel
    def assemble_nodal_force_GN_kernel(self, ):
        dm, nodal_force, nodeEles, nodes, gaussPoints, dsdx, cauchy_stress = ti.static(
            self.dm, self.nodal_force, self.nodeEles, self.nodes, 
            self.ELE.gaussPoints, self.dsdx, self.body.cauchy_stress)
        ### refresh the nodal force before assembling
        for i in nodal_force:
            nodal_force[i] = 0.
        ### begin to assemble
        for node0 in nodes:
            for iele in range(nodeEles[0].n):
                if nodeEles[node0][iele] != -1:
                    ele = nodeEles[node0][iele]

                    ### get the sequence of this node in the element
                    nid = 0  # nid = list(body.np_elements[ele, :]).index(node0)
                    for i in range(self.elements[0].n):
                        if self.elements[ele][i] == node0:
                            nid = i

                    ### assemble stress to the nodal force
                    for igp in range(gaussPoints.shape[0]):
                        dsdx_x_stress = dsdx[ele, igp][nid, :] @ cauchy_stress[ele, igp]
                        for i in range(dm):
                            nodal_force[node0 * dm + i] = \
                            nodal_force[node0 * dm + i] + dsdx_x_stress[i] * self.vol[ele, igp]


    def solve(self, inp: Inp_info, geometric_nonlinear: bool=False, show_newton_steps: bool=False):
        
        if show_newton_steps:
            windowLength = 512
            gui = ti.GUI('show body', res=(windowLength, windowLength))

        print("\033[32;1m now we begin to assemble the sparse matrix \033[0m")
        self.get_dsdx_and_vol()
        self.assemble_sparseMtrx()
        print("\033[32;1m sparse matrix assembling is finished  \033[0m")

        ### use small deformation to compute a step
        self.impose_boundary_condition(inp, geometric_nonlinear=False)
        if geometric_nonlinear == False:
            self.solve_dof(geometric_nonlinear=False)
        
        if geometric_nonlinear:  # large deformation, use newton method
            
            ### recover the rhs
            a_from_b(self.rhs, self.rhs_before_dirichlet)

            ### compute nodal force for large deformation
            self.assemble_nodal_force_GN(); self.assemble_sparseMtrx()
            c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
            self.dirichletBC_forNewtonMethod(inp)
            ini_residual = pre_residual = field_abs_max(self.residual_nodal_force)
            print("\033[40;33;1m initial residual_nodal_force = {} \033[0m".format(ini_residual))
            if show_newton_steps:
                self.body.show2d(gui, disp=self.dof, 
                                 field=self.body.cauchy_stress.to_numpy()[:, :, 0, 0])

            if ini_residual < 1.e-9:
                print("\033[32;1m good! nonlinear converge! \033[0m")
            else:
                newton_loop = -1
                while pre_residual / (ini_residual + 1.e-30) >= 0.01:  # not convergent
                    
                    newton_loop += 1
                    if newton_loop >= 16:
                        break

                    solver = self.solve_dof(geometric_nonlinear=True)  # dofs = dofs - K^(-1) * residual

                    self.assemble_nodal_force_GN(); self.assemble_sparseMtrx()  # use new dofs to compute nodal force
                    ### self.residual_nodal_force = self.nodal_force - self.rhs
                    c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
                    self.dirichletBC_forNewtonMethod(inp)
                    residual = field_abs_max(self.residual_nodal_force)
                    print("\033[40;33;1m newton_loop = {}, residual_nodal_force = {} \033[0m".format(newton_loop, residual))
                    if show_newton_steps:
                        self.body.show2d(gui, disp=self.dof, 
                                        field=self.body.cauchy_stress.to_numpy()[:, :, 0, 0])

                    relax_loop = -1; relaxation = 1.
                    while residual > pre_residual:  # relaxation for Newton's method
                        relax_loop += 1
                        if relax_loop >= 2:
                            break
                        relaxation *= 0.5
                        print("\033[35;1m relaxation = {} \033[0m".format(relaxation))
                        ### self.dof += relaxation * solver.x
                        a_equals_b_plus_c_mul_d(self.dof, self.dof, relaxation, solver.x)
                        self.assemble_nodal_force_GN(); self.assemble_sparseMtrx()  # use new dofs to compute nodal force
                        c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
                        self.dirichletBC_forNewtonMethod(inp)
                        residual = field_abs_max(self.residual_nodal_force)
                        if show_newton_steps:
                            time.sleep(1.)
                            self.body.show2d(gui, disp=self.dof, 
                                            field=self.body.cauchy_stress.to_numpy()[:, :, 0, 0])

                    pre_residual = residual


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    fileName = input("\033[32;1mm please give the .inp format's "
                        "input file path and name: \033[0m")
    ### for example, fileName = ./tests/ellip_membrane_linEle_localVeryFine.inp
    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0])
    ele_type = list(eSets.keys())[0]
    if ele_type[0:3] == "CPS":
        material = Linear_isotropic_planeStress(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])
    elif ele_type[0:3] == "CPE":
        material = Linear_isotropic_planeStrain(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])

    equationSystem = System_of_equations(body, material)
    equationSystem.get_dnatdxs()
    print("\033[35;1m equationSystem.dnatdxs = {}\033[0m".format(equationSystem.dnatdxs))

    equationSystem.solve(inp)
    print("\033[40;33;1m equationSystem.dof = \n{} \033[0m".format(equationSystem.dof.to_numpy()))

    ### show the body
    equationSystem.compute_strain_stress()
    stress_id = int(input("\033[32;1m {} \033[0m".format(
        "which stress do you want to show: \n"
        "0: σxx, 1: σyy, 2: σxy\n"
        "stress index = "
    )))
    stress = equationSystem.body.stresses.to_numpy()[:, :, stress_id]
    print("\033[35;1m maximum stress[{}] = {} MPa \033[0m".format(stress_id, abs(stress).max()))
    
    windowLength = 512
    gui = ti.GUI('show body', res=(windowLength, windowLength))
    while gui.running:
        equationSystem.body.show2d(gui, disp=equationSystem.dof, field=stress)