"""
construct the stiffness matrix
"""
import taichi as ti
import numpy as np
import time; from typing import Tuple, Union
from body import Body
from readInp import *
from conjugateGradientSolver import ConjugateGradientSolver_rowMajor as CG
import user_defined as ud
import tiGadgets as tg
import scipy.sparse as sp
import scipy.sparse.linalg as sl


@ti.data_oriented
class System_of_equations:
    """
    use stiffness matrix and rhs (right hand side) to solve dof (degree of freedom)
    including:
        sparseMtrx: the sparse stiffness matrix utilized to solve the dofs
        rhs: right hand side of the equation system
    """
    def __init__(self, body: Body, material, geometric_nonlinear: bool):
        self.dm = body.dm  # spatial dimension of the system
        self.geometric_nonlinear = geometric_nonlinear
        self.body = body
        self.elements, self.nodes = body.elements, body.nodes 
        
        ### sparseMtrx @ dof = rhs
        self.rhs = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # right hand side of the equation system
        self.dof = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  # degree of freedom that needs to be solved
        
        ### define the element types of the body
        self.ELE = body.ELE

        ### deformation gradient
        self.F = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                                shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        
        ### stress and strain (infinitesimal strain for small deformation and Green strain for large deformation)
        self.cauchy_stress = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))  
        self.strain = ti.Matrix.field(self.dm, self.dm, ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0])) 
        self.mises_stress = ti.field(ti.f64, 
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))   
        self.elsEngDens = ti.field(ti.f64,  # elastic energy density
                        shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))  
        self.elsEng = ti.field(ti.f64, shape=())  # total elastic energy

        ### variables related to geometric nonlinear                              
        self.nodal_force = ti.field(ti.f64, shape=(self.dm * self.body.nodes.shape[0]))  
        self.residual_nodal_force = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))

        ### dsdx (derivative of shape functio with respect to current coordinate), and volume of each guass point
        self.dsdx = ti.Matrix.field(self.elements[0].n, self.dm, ti.f64, 
                                    shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        self.vol = ti.field(ti.f64, shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))
        
        ### constitutive material (e.g., elastic constants) of each element
        self.material = material; self.C = material.C
        self.ddsdde = ti.Matrix.field(n=material.C.n, m=material.C.m, dtype=ti.f64, 
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

        ### get the sparseIJ (index 0 of each row stores the number of effective indexes in this row)
        body.get_coElement_nodes()
        maxLen = max(len(body.coElement_nodes[node]) for node in range(body.nodes.shape[0]))
        self.sparseIJ = ti.Vector.field(maxLen * self.dm + 1, ti.i32, 
            shape=(self.nodes.shape[0] * self.dm, ))
        sparseIJ = -np.ones((self.nodes.shape[0] * self.dm, maxLen * self.dm + 1))
        for node0 in range(self.nodes.shape[0]):
            js = [node1 * self.dm + i for node1 in body.coElement_nodes[node0] for i in range(self.dm)]
            for i in range(self.dm):
                sparseIJ[node0 * self.dm + i][1:len(js)+1] = js[:]
                sparseIJ[node0 * self.dm + i][0] = len(js)
        self.sparseIJ.from_numpy(sparseIJ)

        ### init the row major form of sparse matrix
        print("\033[32;1m shape of the sparseIJ is {} \033[0m".format(self.sparseIJ.shape))
        self.sparseMtrx_rowMajor = ti.Vector.field(self.sparseIJ[0].n - 1, ti.f64, 
            shape=(self.sparseIJ.shape[0], ))
        self.du = ti.field(ti.f64, shape=(self.nodes.shape[0] * self.dm))

        ### sparse matrix (indexes and elements) prepared for scipy
        self.rows, self.cols = [], []  # indexes of rows and coloums
        sparseIJ = self.sparseIJ.to_numpy()
        for i in range(sparseIJ.shape[0]):
            for j_ in range(sparseIJ[i, 0]):
                j = sparseIJ[i, j_ + 1]
                self.rows.append(i)
                self.cols.append(j)
        self.rows, self.cols = np.array(self.rows), np.array(self.cols)
        self.sparseIJ_np = sparseIJ
        self.K = np.zeros(shape=(self.rows.shape[0],), dtype=np.float64)

        ### initial variables related to time increments
        self.time0 = 0.; self.time1 = 0.
        self.dt = 0.
        # degree of freedom at last time step
        self.dof_old = ti.field(ti.f64, shape=(body.nodes.shape[0] * body.dm, ))  

        ### some variables for print
        self.compiled = False  # indicate whether the assemble_sparseMtrx has been compiled
        # a field for visualization, you can visualize some idexes of stress or strain
        self.visualize_field = ti.field(ti.f64,  
            shape=(self.elements.shape[0], self.ELE.gaussPoints.shape[0]))   
        self.nodal_vals = ti.Vector.field(self.elements[0].n, ti.f64, 
            shape=(self.elements.shape[0],)) # visualization of nodal strain or stress


    @ti.kernel
    def ddsdde_init(self, ):
        """get the ddsdde (material Jacobian, ∂Δσ/∂Δε) 
           at each integration point of each element"""
        for ele, igp in self.ddsdde:
            self.ddsdde[ele, igp] = self.C


    @ti.kernel 
    def get_dsdx_and_vol(self, ):
        """update dsdx (∇N) and volume before assemble stiffness matrix"""
        dm, elements, nodes, dof, gaussPoints  = ti.static(
            self.dm, self.elements, self.nodes, self.dof, self.ELE.gaussPoints)
        for ele in self.elements:
            localNodes = ti.Matrix.zero(ti.f64, elements[0].n, dm)
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


    def assemble_sparseMtrx(self, ):
        tiMatrixShape = self.elements[0].n * self.dm
        if tiMatrixShape <= 10:  # involve small ti.Matrix()
            self.assemble_stiffnessMtrx()
        else:  # involve large ti.Matrix(), use faster version to save compile time
            self.assemble_stiffnessMtrx_faster() 


    @ti.kernel
    def assemble_stiffnessMtrx(self, ):
        """assemble sitffness matrix, parallel by integration points, 
           update self.dsdx (∇N) and self.vol before assemble this"""
        dm, sparseMtrx_rowMajor, elements, ddsdde, vol, dsdx = ti.static(
            self.body.dm, self.sparseMtrx_rowMajor, self.elements, self.ddsdde, self.vol, self.dsdx)
        
        sparseMtrx_rowMajor.fill(0.)
        for ele, igp in vol:  # parallel by integration points
            strain = self.ELE.strainMtrx(dsdx[ele, igp])  # strain B(∇N), B·u = ε
            stress_voigt = ddsdde[ele, igp] @ strain  # stress = C·B
            bcb = strain.transpose() @ stress_voigt  # stiffness, BT·C·B
            
            ### integrate to the large sparse matrix
            for node in range(elements[ele].n):
                node0 = elements[ele][node]  # global node index
                Js = ti.Vector([x + i for x in elements[ele]*dm for i in range(dm)])
                for i_local in range(dm):
                    i_global = node0 * dm + i_local
                    for j_local in range(Js.n):
                        j_global = Js[j_local]
                        j = self.sparseMatrix_get_j(i_global, j_global)
                        ### atomic add to related node
                        sparseMtrx_rowMajor[i_global][j] += \
                            bcb[node*dm+i_local, j_local] * vol[ele, igp]


    @ti.kernel
    def assemble_stiffnessMtrx_faster(self, ):
        """assemble sitffness matrix, parallel by elements, 
           compiles faster by ∇N·C·B instead of BT·C·B
           (also, update self.dsdx (∇N) and self.vol before assemble this)"""
        dm, sparseMtrx_rowMajor, elements, ddsdde, vol = ti.static(
            self.body.dm, self.sparseMtrx_rowMajor, self.elements, self.ddsdde, self.vol)
        
        sparseMtrx_rowMajor.fill(0.)
        for ele, igp in vol:  # parallel by integration points
            dsdx = self.dsdx[ele, igp]  # ∇N, the updated grad of shape function
            strain = self.ELE.strainMtrx(dsdx)  # strain B(∇N), B·u = ε
            stress_voigt = ddsdde[ele, igp] @ strain  # stress = C·B

            ### integrate to the large sparse matrix
            for node in range(elements[ele].n):
                ### dsdx mutiplies the stress, ∇N·C·B, compile faster than BT·C·B
                dsdx_x_stress = tg.vec_mul_voigtMtrx(dsdx[node, :], stress_voigt)
                node0 = elements[ele][node]  # global node index
                Js = ti.Vector([x + i for x in elements[ele]*dm for i in range(dm)])
                for i_local in range(dm):
                    i_global = node0 * dm + i_local
                    for j_local in range(Js.n):
                        j_global = Js[j_local]
                        j = self.sparseMatrix_get_j(i_global, j_global)
                        ### atomic add to related node
                        sparseMtrx_rowMajor[i_global][j] += \
                            dsdx_x_stress[i_local, j_local] * vol[ele, igp] 
    

    def solve_by_scipy(self, ):
        ### conver some field to np.array
        sparseIJ = self.sparseIJ_np
        sparseMtrx_rowMajor = self.sparseMtrx_rowMajor.to_numpy()

        ### fetch the sparse matrix
        id = -1
        for i in range(sparseIJ.shape[0]):
            for j_ in range(sparseIJ[i, 0]):
                id += 1
                self.K[id] = sparseMtrx_rowMajor[i, j_]
        time0 = time.time()
        K = sp.csr_matrix((self.K, (self.rows, self.cols)), 
                          shape=(sparseIJ.shape[0], 
                                 sparseIJ.shape[0]), dtype=np.float64)

        ### solve the sparse matrix equation AX = B
        if not self.geometric_nonlinear:
            self.du.from_numpy(sl.spsolve(K, self.rhs.to_numpy()))
        else:
            self.du.from_numpy(sl.spsolve(K, self.residual_nodal_force.to_numpy()))
        
        time1 = time.time()
        print(f"\033[32;1m assuming time for sparse matrix solving "
              f"in scipy is {time1 - time0} s \033[0m")

        if not self.geometric_nonlinear:
            self.dof = self.du
        else:
            ### self.dof = self.dof - solver.x (in Newton's method)
            tg.c_equals_a_minus_b(self.dof, self.dof, self.du)
        
        return self.du
    

    def solve_by_CG(self):
        if not hasattr (self, "PCG"):
            if not self.geometric_nonlinear:
                self.PCG = CG(spm=self.sparseMtrx_rowMajor, sparseIJ=self.sparseIJ, b=self.rhs)
            else:
                self.PCG = CG(spm=self.sparseMtrx_rowMajor, sparseIJ=self.sparseIJ, b=self.residual_nodal_force)
        self.PCG.re_init() 
        self.PCG.solve()

        if not self.geometric_nonlinear:
            self.dof = self.PCG.x
        else:
            ### self.dof = self.dof - solver.x (in Newton's method)
            tg.c_equals_a_minus_b(self.dof, self.dof, self.PCG.x)
        
        return self.PCG.x
    

    def solve_dof(self, ):
        if self.dof.shape[0] < 1e5:  # critical matrix size
            return self.solve_by_scipy()  # direct method
        else:
            return self.solve_by_CG()  # CG (can parallel by gpu)


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
            for j0 in range(self.sparseIJ[i_global][0]):
                j_global = self.sparseIJ[i_global][j0 + 1]
                ### use the symmetric property of the sparse matrix, find sparseMtrx[j_global, i_global]
                i0 = self.sparseMatrix_get_j(j_global, i_global)
                self.rhs[j_global] = self.rhs[j_global] - sval * self.sparseMtrx_rowMajor[j_global][i0]
            self.rhs[i_global] = sval

            ### modify the sparse matrix
            for j0 in range(self.sparseIJ[i_global][0]):
                j_global = self.sparseIJ[i_global][j0 + 1]
                self.sparseMtrx_rowMajor[i_global][j0] = 0.
                i0 = self.sparseMatrix_get_j(j_global, i_global)
                self.sparseMtrx_rowMajor[j_global][i0] = 0.
            i0 = self.sparseMatrix_get_j(i_global, i_global)
            self.sparseMtrx_rowMajor[i_global][i0] = 1.


    def dirichletBC_forNewtonMethod(self, dirichletBCs):
        for dirichletBC in dirichletBCs:
            self.dirichletBC_dof(dirichletBC["node_set"], dirichletBC["dof"], 
                                dirichletBC["val"], dirichletBC["user"], self.time1)
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
        
        ### impose dirichlet BC at residual force and sparse matrix
        for node_ in nodeSet:
            node = nodeSet[node_]
            i_global = node * self.dm + dm_specified

            ### modify the residual force
            self.residual_nodal_force[i_global] = 0.

            ### modify the sparse matrix (i.e., the Jacobian)
            for j0 in range(self.sparseIJ[i_global][0]):
                j_global = self.sparseIJ[i_global][j0 + 1]
                self.sparseMtrx_rowMajor[i_global][j0] = 0.
                i0 = self.sparseMatrix_get_j(j_global, i_global)
                self.sparseMtrx_rowMajor[j_global][i0] = 0.
            i0 = self.sparseMatrix_get_j(i_global, i_global)
            self.sparseMtrx_rowMajor[i_global][i0] = 1.
    

    def dirichletBC_dof(self, 
                    nodeSet: ti.template(), dm_specified: int,  # the specified dimendion of dirichlet BC 
                    sval: float,  # specific value of dirichlet boundary condition
                    user: bool,  # ture means using user defined boundary condition
                    time: float, 
                    ):
        if not user:
            self.dirichletBC_val(nodeSet, dm_specified, sval)
        else:
            ud.user_dirichletBC(
                self.dof, nodeSet, self.dm, dm_specified, self.nodes, time)
    

    @ti.kernel
    def dirichletBC_val(self, 
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
        self.rhs.fill(0.)  # refresh the right hand side before apply Neumann BC
        
        for facet in load_facets:
            ele = body.boundary[facet]
            for node0 in facet:  # traction force of this facet applies to node0

                ### obtain the facet normals on the free-load boundary 
                ###     (points to the outside of the element)
                localNodes = np.array([body.np_nodes[node, :] for node in body.np_elements[ele, :]])
                eleNodesList = body.np_elements[ele, :].tolist()
                localFacet = [eleNodesList.index(i) for i in facet]
                for integId in range(ELE.integPointNum_eachFacet):
                    normal_vector, area_x_weight = ELE.globalNormal(nodes=localNodes, 
                                                                    facet=localFacet, 
                                                                    integPointId=integId)
                    ### get the flux, which can also be interpreted as traction force
                    if len(load_dir) == 0: 
                        flux = load_val * normal_vector * area_x_weight
                    else:
                        flux = load_val * load_dir * area_x_weight   

                    ### get the sequence of this node in the element
                    natCoo = ELE.facet_natural_coos[tuple(sorted(localFacet))][integId]
                    nid = list(body.np_elements[ele, :]).index(node0)
                    shapeVal = ELE.shapeFunc_pyscope(natCoo)[nid] 

                    for i in range(self.dm):
                        self.rhs[node0*self.dm + i] += flux[i] * shapeVal

    
    @ti.func
    def sparseMatrix_get_j(self, i_global, j_global):
        j_local = 0
        for j in range(self.sparseIJ[i_global][0]):
            if self.sparseIJ[i_global][j + 1] == j_global:
                j_local = j
        return j_local
    

    def check_sparseIJ(self, ):
        """check whether indexes repeatly appear in sparseIJ"""
        for i in range(self.sparseIJ.shape[0]):
            js = set()
            for j0 in range(self.sparseIJ[i][0]):
                j = self.sparseIJ[i][j0 + 1]
                if j != -1:
                    if j in js:
                        print("\033[31;1m error, {}, {} appear repeatly \033[0m".format(i, j))
                    else:
                        js.add(j)


    def compute_strain_stress(self, ):
        ### get the strain
        self.get_deformation_gradient()
        if not self.geometric_nonlinear:
            self.get_strain_smallDeformation()
        else:
            self.get_strain_largeDeformation()
        ### get the stress
        if not self.geometric_nonlinear:
            self.material.constitutiveOfSmallDeform(self.F, self.cauchy_stress, self.ddsdde)
        else:
            pass  # stress has been computed for geometric nonlinear case
        ### compute mises stress
        if self.material.type == "planeStrain":
            self.get_mises_stress_planeStrain()
        elif self.material.type == "planeStress":
            self.get_mises_stress_planeStress()
        else:
            self.get_mises_stress_3d()
    

    @ti.kernel 
    def get_mises_stress_planeStress(self, ):
        eye = ti.Matrix([[1., 0., 0.], 
                         [0., 1., 0.], 
                         [0., 0., 1.]])
        for ele in self.elements:
            for igp in range(self.ELE.gaussPoints.shape[0]):
                stress_2d = self.cauchy_stress[ele, igp]
                stress = ti.Matrix([ 
                    [stress_2d[0, 0], stress_2d[0, 1], 0.], 
                    [stress_2d[1, 0], stress_2d[1, 1], 0.],
                    [0., 0., 0.],
                ])
                deviatoric_stress = stress - eye * stress.trace() / 3.
                self.mises_stress[ele, igp] = (3./2. * (deviatoric_stress * deviatoric_stress).sum())**0.5
    

    @ti.kernel 
    def get_mises_stress_planeStrain(self, ):
        nu = self.material.poisson_ratio
        eye = ti.Matrix([[1., 0., 0.], 
                         [0., 1., 0.], 
                         [0., 0., 1.]])
        for ele in self.elements:
            for igp in range(self.ELE.gaussPoints.shape[0]):
                stress_2d = self.cauchy_stress[ele, igp]
                stress = ti.Matrix([ 
                    [stress_2d[0, 0], stress_2d[0, 1], 0.], 
                    [stress_2d[1, 0], stress_2d[1, 1], 0.],
                    [0., 0., nu * (stress_2d[0, 0] + stress_2d[1, 1])],
                ])
                deviatoric_stress = stress - eye * stress.trace() / 3.
                self.mises_stress[ele, igp] = (3./2. * (deviatoric_stress * deviatoric_stress).sum())**0.5


    @ti.kernel
    def get_mises_stress_3d(self, ):
        eye = ti.Matrix([[1., 0., 0.], 
                         [0., 1., 0.], 
                         [0., 0., 1.]])
        for ele in self.elements:
            for igp in range(self.ELE.gaussPoints.shape[0]):
                stress = self.cauchy_stress[ele, igp]
                deviatoric_stress = stress - eye * stress.trace() / 3.
                self.mises_stress[ele, igp] = (3./2. * (deviatoric_stress * deviatoric_stress).sum())**0.5


    def impose_boundary_condition(self, boundary_conditions: dict):
        ### =========== apply the boundary condition ===========
        neumannBCs = boundary_conditions["neumannBCs"]
        dirichletBCs = boundary_conditions["dirichletBCs"]
        
        ### first, apply Neumann BC
        for neumannBC in neumannBCs:
            if "direction" in neumannBC:
                self.neumannBC(neumannBC["face_set"], 
                                load_val=neumannBC["traction"], 
                                load_dir=neumannBC["direction"])
            else:
                self.neumannBC(neumannBC["face_set"], 
                                load_val=neumannBC["traction"])
        
        ### then, apply Dirichlet BC
        if self.geometric_nonlinear == False:
            for dirichletBC in dirichletBCs:
                self.dirichletBC_linearEquations(
                        dirichletBC["node_set"], dirichletBC["dof"], 
                        dirichletBC["val"])
        else:  # large deformation (geometric nonlinear), dirichlet BC is imposed latter at Newton's method
            for dirichletBC in dirichletBCs:
                self.dirichletBC_dof(
                        dirichletBC["node_set"], dirichletBC["dof"], 
                        dirichletBC["val"], dirichletBC["user"], self.time1)


    @ti.kernel 
    def get_deformation_gradient(self, ):
        """get the deformation gradient of each integration point"""
        dm, elements, nodes, gaussPoints = ti.static(
            self.dm, self.elements, self.nodes, self.ELE.gaussPoints)
        eye = ti.Matrix.zero(ti.f64, dm, dm)
        for i in range(eye.n):
            eye[i, i] = 1.
        for ele in elements:
            local_us = ti.Matrix.zero(ti.f64, elements[0].n, dm)
            local_nodes = ti.Matrix.zero(ti.f64, elements[0].n, dm)
            for node_ in range(elements[0].n):
                node = elements[ele][node_]
                for j in ti.static(range(dm)):
                    local_us[node_, j] = self.dof[node * dm + j]  # e.g., displacement
                    local_nodes[node_, j] = nodes[node][j]
            for igp in range(gaussPoints.shape[0]):
                gp = gaussPoints[igp]
                dsdn = self.ELE.dshape_dnat(gp)  # derivative of shape function with respect to natural coodinates
                dnatdX = (local_nodes.transpose() @ dsdn).inverse()  # deformation gradient refers to initial configuration
                dsdX = dsdn @ dnatdX  # strain = (u_node1, u_node2, u_node3) * dsdx
                dudX = local_us.transpose() @ dsdX
                
                ### get the deformation gradient, F = I + dudX
                self.F[ele, igp] = dudX + eye
    

    @ti.kernel 
    def get_strain_smallDeformation(self, ):
        """get the strain directly according to deformation gradient
           the strain here is just for visulization, not for constitutive"""
        dm, elements, gaussPoints, strain = ti.static(
            self.dm, self.elements, self.ELE.gaussPoints, self.strain)
        eye = ti.Matrix.zero(ti.f64, dm, dm)
        for i in range(eye.n):
            eye[i, i] = 1.
        
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]
                strain[ele, igp] = (F + F.transpose()) / 2. - eye
        

    @ti.kernel 
    def get_strain_largeDeformation(self, ):
        """get the strain directly according to deformation gradient
           the strain here is just for visulization, not for constitutive
           Green strain for large deformation"""
        dm, elements, gaussPoints, strain = ti.static(
            self.dm, self.elements, self.ELE.gaussPoints, self.strain)
        eye = ti.Matrix.zero(ti.f64, dm, dm)
        for i in range(eye.n):
            eye[i, i] = 1.
        
        for ele in elements:
            for igp in range(gaussPoints.shape[0]):
                F = self.F[ele, igp]
                strain[ele, igp] = (F.transpose() @ F - eye) / 2.


    def get_elasEng(self, ):
        """get elatic energy (density and total energy)"""
        self.get_deformation_gradient()
        self.get_elasEng_kernel()
    
    @ti.kernel
    def get_elasEng_kernel(self, ):
        """first, get elatic energy density"""
        for I in ti.grouped(self.elsEngDens):
            self.elsEngDens[I] = self.material.elasticEnergyDensity(
                self.F[I])
        """then, get total elastic energy"""
        self.elsEng[None] = 0.
        for I in ti.grouped(self.elsEngDens):
            self.elsEng[None] += self.elsEngDens[I] * self.vol[I]


    def assemble_nodal_force_GN(self, ):
        """assemble the nodal force for GN (geometric nonlinear)"""
        ### get all stresses at integration points by constitutive, modified latter by automatically change consititutive
        self.get_deformation_gradient()
        self.material.constitutiveOfLargeDeform(self.F, self.cauchy_stress, self.ddsdde)
        ### get dsdx and vol
        self.get_dsdx_and_vol()
        ### assemble to nodal force
        self.assemble_nodal_force_GN_kernel()

    
    @ti.kernel
    def assemble_nodal_force_GN_kernel(self, ):
        dm, nodal_force, nodeEles, nodes, gaussPoints, dsdx, cauchy_stress = ti.static(
            self.dm, self.nodal_force, self.nodeEles, self.nodes, 
            self.ELE.gaussPoints, self.dsdx, self.cauchy_stress)
        ### refresh the nodal force before assembling
        for i in nodal_force:
            nodal_force[i] = 0.
        ### begin to assemble
        for node0 in nodes:
            for iele in range(nodeEles[0].n):
                if nodeEles[node0][iele] != -1:
                    ele = nodeEles[node0][iele]

                    ### get the sequence of this node in the element
                    nid = tg.get_index_ti(self.elements[ele], node0)
                    if nid == -1:
                        print("\033[31;1m Error, index not found. nid = -1 \033[0m")

                    ### assemble stress to the nodal force
                    for igp in range(gaussPoints.shape[0]):
                        dsdx_x_stress = dsdx[ele, igp][nid, :] @ cauchy_stress[ele, igp]
                        for i in range(dm):
                            nodal_force[node0 * dm + i] = \
                            nodal_force[node0 * dm + i] + dsdx_x_stress[i] * self.vol[ele, igp]


    def solve(self, inp: Inp_info, show_newton_steps: bool=False, save2path: str=None):
        """solved by multiple time increments, each increment calls slove_inc()"""
        max_inc = inp.time_incs["max_inc"]
        min_inc = inp.time_incs["min_inc"]
        max_time = inp.time_incs["max_time"]
        self.dt = inp.time_incs["ini_inc"]
        
        neumannBCs = copy.deepcopy(inp.neumann_bc_info)
        dirichletBCs = copy.deepcopy(inp.dirichlet_bc_info)
        for dirichletBC in dirichletBCs:
            node_set = ti.field(ti.i32, shape=(len(dirichletBC["node_set"])))
            node_set.from_numpy(np.array([*dirichletBC["node_set"]]))
            dirichletBC["node_set"] = node_set  # replace the original node set by field
        boundary_conditions = {"neumannBCs": neumannBCs, "dirichletBCs": dirichletBCs}

        ### whether show the body during time step and Newton's step
        if self.geometric_nonlinear:
            windowLength = 512
            window = ti.ui.Window('show body', (windowLength, windowLength))
        else: window = None

        ### visualize and image IO at the initial step
        if self.geometric_nonlinear:
            fileName = f"{save2path}_time{self.time0:.4f}.png" if save2path else None
            self.show_window(window, fileName)
            if show_newton_steps:
                fileName = self.write_image_name(save2path, 0, 0)
                self.show_window(window, fileName)

        ### now start the time increments
        kinc = -1  # the number of time increment
        while self.time1 < max_time:
            kinc += 1
            self.time1 = min(self.time0 + self.dt, max_time)
            print("\033[40;33;1m >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                                ">>>>> kinc = {}, time0 = {}, dt = {} \033[0m".format(kinc, self.time0, self.dt))
            load_ratio = self.time1 / max_time
            ### set the boundary according to load ratio
            for id, neumannBC in enumerate(neumannBCs):
                neumannBC["traction"] = inp.neumann_bc_info[id]["traction"] * load_ratio
            for id, dirichletBC in enumerate(dirichletBCs):
                dirichletBC["val"] = inp.dirichlet_bc_info[id]["val"] * load_ratio
            ### advance a time increment
            converged, newton_loop = self.advance_inc(inp, boundary_conditions, 
                                                      show_newton_steps, save2path, window)  # update self.dof
            if not converged:
                self.time1 = self.time0
                self.dt /= 4.
                self.dof.copy_from(self.dof_old)
                kinc -= 1
                if self.dt < min_inc:
                    print("\033[31;1m allowable minimum dt is reached, "
                          "Newton's method not converges, solution is not found. \033[0m")
                    break
                continue
            ### increast dt if fast convergence occurs in previous dt
            if newton_loop <= 8:
                self.dt = min(self.dt * 1.5, max_inc)
            self.dof_old.copy_from(self.dof)
            self.time0 = self.time1

            ### visualize and image IO
            if self.geometric_nonlinear:
                fileName = f"{save2path}_time{self.time1:.4f}.png" if save2path else None
                self.show_window(window, fileName)
    
    
    def advance_inc(self, inp: Inp_info, boundary_conditions: dict, 
                    show_newton_steps: bool=False, save2path: str=None, 
                    window: Union[ti.ui.Window, ti.GUI]=None, 
                  ) -> Tuple[bool, int]:

        def inside_relaxation():
            self.assemble_nodal_force_GN(); self.assemble_stiffnessMtrx()  # use new dofs to compute nodal force
            tg.c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
            self.dirichletBC_forNewtonMethod(boundary_conditions["dirichletBCs"])
            residual = tg.field_norm(self.residual_nodal_force)
            print("\033[32;1m residual = {} \033[0m".format(residual))
            if show_newton_steps:
                fileName = self.write_image_name(save2path, newton_loop+1, relax_loop+1)
                self.show_window(window, fileName)
            return residual
        

        """solve at each time increment"""
        geometric_nonlinear = inp.geometric_nonlinear
        print("\033[35;1m >>> geometric nonlinear is {}. \033[0m".format(
            {False: "off", True: "on"}[geometric_nonlinear]))

        print("\033[32;1m now we begin to assemble the sparse matrix \033[0m"); time0 = time.time()
        self.get_dsdx_and_vol()
        self.assemble_stiffnessMtrx()
        print("\033[32;1m sparse matrix assembling is finished  \033[0m"); time1 = time.time()
        if not self.compiled:
            self.compiled = True
            print("\033[35;1m assemble_sparseMtrx's compiling time is {} s\033[0m".format(time1 - time0))
        else:
            print("time for assemble is {} s".format(time1 - time0))

        ### impost boundary condition at the initial state
        self.impose_boundary_condition(boundary_conditions)
        
        if geometric_nonlinear == False:  # small deformation
            self.solve_dof()
            return True, 0
        
        else:  # large deformation, use newton method
            
            ### compute nodal force for large deformation
            self.assemble_nodal_force_GN(); self.assemble_stiffnessMtrx()
            tg.c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
            self.dirichletBC_forNewtonMethod(boundary_conditions["dirichletBCs"])
            pre_residual = tg.field_norm(self.residual_nodal_force)
            if not hasattr(self, "ini_residual"):
                self.ini_residual = pre_residual
            print("\033[40;33;1m initial residual_nodal_force = {} \033[0m".format(self.ini_residual))
            if show_newton_steps:
                fileName = self.write_image_name(save2path, 1, 1)
                self.show_window(window, fileName)

            if self.ini_residual < 1.e-9:
                print("\033[32;1m good! nonlinear converge! \033[0m")
            else:
                newton_loop = -1
                while pre_residual / (self.ini_residual + 1.e-30) >= 0.01:  # not convergent
                    
                    newton_loop += 1
                    if newton_loop >= 24:
                        return False, newton_loop  # Newton's method has not converged

                    du = self.solve_dof()  # dofs = dofs - K^(-1) * residual

                    self.assemble_nodal_force_GN(); self.assemble_stiffnessMtrx()  # use new dofs to compute nodal force
                    ### self.residual_nodal_force = self.nodal_force - self.rhs
                    tg.c_equals_a_minus_b(self.residual_nodal_force, self.nodal_force, self.rhs)
                    self.dirichletBC_forNewtonMethod(boundary_conditions["dirichletBCs"])
                    residual = tg.field_norm(self.residual_nodal_force)
                    if np.isnan(residual):
                        print("NaN occurs, automatically recompute with smaller time step")
                        return False, newton_loop
                    print("\033[40;33;1m newton_loop = {}, residual_nodal_force = {} \033[0m".format(newton_loop, residual))
                    if show_newton_steps:
                        fileName = self.write_image_name(save2path, newton_loop+1, 1)
                        self.show_window(window, fileName)

                    ### boost Newton's method by going a larger step if residual force is declining
                    relax_loop = -1; relaxation = 1.
                    relaxation = 1.  #  further_step_ratio
                    while 0.1 * pre_residual < residual < pre_residual:  # when residual declines, go on this direction further
                        new_residual = residual
                        relax_loop += 1
                        if relax_loop >= 10:
                            break
                        print("\033[35;1m further_step_ratio = {} \033[0m".format(relaxation))
                        ### self.dof -= relaxation * du
                        tg.a_equals_b_plus_c_mul_d(self.dof, self.dof, -relaxation, du)
                        residual = inside_relaxation()
                        if residual > new_residual:
                            tg.a_equals_b_plus_c_mul_d(self.dof, self.dof, +relaxation, du)
                            residual = inside_relaxation()
                            relaxation *= 0.5
                    
                    ### relaxation for Newton's method when residual gets bigger
                    relax_loop = -1; relaxation = 0.5
                    while residual > pre_residual:
                        relax_loop += 1
                        if relax_loop >= 2:
                            break
                        print("\033[35;1m relaxation = {} \033[0m".format(relaxation))
                        ### self.dof += (1. - relaxation) * du, i.e., recover dof, then update with relaxation  
                        tg.a_equals_b_plus_c_mul_d(self.dof, self.dof, (1. - relaxation), du)
                        tg.field_multiply(du, relaxation)
                        residual = inside_relaxation()

                    pre_residual = residual
            return True, newton_loop  # Newton's method converges


    def show_window(self, window, save2path: str=None):
        self.compute_strain_stress()
        self.ELE.extrapolate(self.mises_stress, self.nodal_vals)
        self.body.show(window, self.dof, self.nodal_vals, save2path)


    def write_image_name(self, save2path:str, newton_loop:int, relax_loop:int):
        writeFrequency = 2
        if not save2path:
            writeImage = False
        else:
            if newton_loop % writeFrequency == 0 and relax_loop % writeFrequency == 0:
                writeImage = True
            else:
                writeImage = False
        if writeImage:
            fileName = f"{save2path}_{self.time0:.4f}_{newton_loop}_{relax_loop}_.png"
        else:
            fileName = None
        return fileName
        

if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True, default_fp=ti.f64)
    fileName = input("\033[32;1m please give the .inp format's "
                        "input file path and name: \033[0m")
    ### for example, fileName = ./tests/elliptic_membrane/element_linear/ellip_membrane_linEle_localVeryFine.inp
    inp = Inp_info(fileName)
    nodes, eSets = inp.nodes, inp.eSets
    body = Body(nodes=nodes, elements=list(eSets.values())[0])
    ele_type = list(eSets.keys())[0]
    if ele_type[0:3] == "CPS":
        material = LinearIsotropicPlaneStress(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])
    elif ele_type[0:3] == "CPE":
        material = LinearIsotropicPlaneStrain(modulus=inp.materials["Elastic"][0], 
                                                poisson_ratio=inp.materials["Elastic"][1])

    equationSystem = System_of_equations(body, material, inp.geometric_nonlinear)

    equationSystem.solve(inp)
    print("\033[40;33;1m equationSystem.dof = \n{} \033[0m".format(equationSystem.dof.to_numpy()))

    ### show the body
    equationSystem.compute_strain_stress()
    stress = equationSystem.mises_stress.to_numpy()
    print("\033[35;1m maximum mises stress = {} MPa \033[0m".format(abs(stress).max()))
    print("\033[40;33;1m max dof (disp) = {} \033[0m".format(tg.field_abs_max(equationSystem.dof)))
    
    windowLength = 512
    gui = ti.GUI('show body', res=(windowLength, windowLength))
    while gui.running:
        equationSystem.body.show2d(gui, disp=equationSystem.dof, field=stress)